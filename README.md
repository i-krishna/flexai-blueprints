# blueprints

Debugging scripts from https://github.com/flexaihq/blueprints 

# Why you can't just use GPUs for everything
A natural question: if GPUs are so fast, why not run Spark on them too?
You could — NVIDIA has RAPIDS/cuDF which does DataFrame operations on GPUs. But most data engineering work is irregular: reading files of different sizes, handling messy strings, joining tables with unpredictable cardinality, skipping partitions based on filters. GPUs hate irregular work — they're designed for lockstep, uniform operations where all cores do exactly the same thing at the same time.
Also, GPU memory is tiny relative to data lake scale. An A100 has 80GB. A typical Silver table might be 50TB. You'd spend all your time shuffling data in and out of GPU memory — completely defeating the purpose.
Conversely, you can't run a transformer model efficiently on a Spark CPU cluster — the matrix multiplications would take weeks instead of hours.
They are purpose-built for genuinely different problems, which is why modern AI + data platforms use both.

<img width="469" height="338" alt="image" src="https://github.com/user-attachments/assets/f3c576ef-c932-4abf-8b77-9320c897c03c" />

What MIG actually is — physically
An A100 or H100 GPU is not one monolithic chip. Internally it's built from GPU Memory Slices and Compute Slices (called SM — Streaming Multiprocessors). MIG lets you cut these physical slices into independent instances.

On an A100 (80GB):
A100 Physical GPU
├── 108 Streaming Multiprocessors (SMs)
├── 80GB HBM2e memory
└── 6 memory controllers

MIG partitioning options:
┌─────────────────────────────────────────────┐
│  1x  MIG 7g.80gb  → full GPU  (1 instance)  │
├─────────────┬───────────────────────────────┤
│  2x  MIG 3g.40gb  → half each (2 instances) │
├──────┬──────┬──────┬────────────────────────┤
│  7x  MIG 1g.10gb  → 1/7 each (7 instances)  │
└──────┴──────┴──────┴────────────────────────┘

Each MIG instance gets:

A dedicated slice of SMs (compute)
A dedicated slice of HBM memory (no sharing)
A dedicated memory controller (no bandwidth contention)
Its own L2 cache partition

This is hardware-level isolation — not virtualisation, not software partitioning. The silicon is physically divided.

# How isolation is enforced — the three layers

## Layer 1: Hardware (silicon-level)
NVIDIA's MIG architecture enforces isolation in the hardware itself. Each instance has its own:
MIG Instance A (Customer 1)        MIG Instance B (Customer 2)
├── SMs 0-13  (dedicated)          ├── SMs 14-27 (dedicated)
├── 10GB HBM  (dedicated)          ├── 10GB HBM  (dedicated)
├── Memory controller 0            ├── Memory controller 1
└── L2 cache partition A           └── L2 cache partition B

              ↑
    Hardware MMU (Memory Management Unit)
    enforces that SM 0-13 can NEVER
    address memory belonging to partition B
    This is not software — it's circuit-level

A workload running in MIG instance A literally cannot generate a memory address that falls inside MIG instance B's memory range. The hardware Memory Management Unit rejects it at the silicon level — no software bug in your code, no misconfigured container, no malicious tenant can cross this boundary.

## Layer 2: OS / Driver (NVIDIA driver + CUDA)
The NVIDIA driver exposes each MIG instance as a separate device:
bash# Without MIG — one device
$ nvidia-smi
GPU 0: A100-SXM4-80GB

# With MIG enabled — 7 separate devices
$ nvidia-smi -L
GPU 0: A100-SXM4-80GB
  MIG 1g.10gb  Device 0: (UUID: MIG-xxxxxxx-0)
  MIG 1g.10gb  Device 1: (UUID: MIG-xxxxxxx-1)
  MIG 1g.10gb  Device 2: (UUID: MIG-xxxxxxx-2)
  MIG 1g.10gb  Device 3: (UUID: MIG-xxxxxxx-3)
  MIG 1g.10gb  Device 4: (UUID: MIG-xxxxxxx-4)
  MIG 1g.10gb  Device 5: (UUID: MIG-xxxxxxx-5)
  MIG 1g.10gb  Device 6: (UUID: MIG-xxxxxxx-6)
From the OS perspective, these are completely separate GPUs. A process assigned to Device 0 cannot see Device 1 — the driver enforces this at the device file level (/dev/nvidia0 vs /dev/nvidia1).

## Layer 3: Kubernetes / Container (orchestration)
In a production multi-tenant platform, you run MIG instances as Kubernetes resources using the NVIDIA Device Plugin and MIG Manager:
yaml

### Customer 1's inference pod
apiVersion: v1
kind: Pod
metadata:
  name: customer-1-inference
  namespace: tenant-customer-1      # ← separate namespace per tenant
spec:
  containers:
  - name: model-server
    image: triton-inference:latest
    resources:
      limits:
        nvidia.com/mig-1g.10gb: 1  # ← gets exactly one MIG slice
    env:
    - name: CUDA_VISIBLE_DEVICES
      value: "MIG-xxxxxxx-0"        # ← pinned to specific instance
---
### Customer 2's inference pod
apiVersion: v1
kind: Pod
metadata:
  name: customer-2-inference
  namespace: tenant-customer-2      # ← completely separate namespace
spec:
  containers:
  - name: model-server
    image: triton-inference:latest
    resources:
      limits:
        nvidia.com/mig-1g.10gb: 1  # ← gets a different MIG slice
    env:
    - name: CUDA_VISIBLE_DEVICES
      value: "MIG-xxxxxxx-1"        # ← pinned to different instance

CUDA_VISIBLE_DEVICES is the key environment variable — it tells the CUDA runtime which devices exist from this process's perspective. Customer 1's container literally cannot see any MIG instance other than MIG-xxxxxxx-0. Even if a bug in their code tried to access cuda:1, CUDA returns no device found.



