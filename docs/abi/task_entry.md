# Task Entry ABI - Thing-OS

This document defines the contract for task entry in Thing-OS, ensuring consistent and predictable startup state for both kernel and user tasks.

## Registers

Upon entry, the first argument (`ARG0`) is passed in an architecture-specific register:

| Architecture | Register |
|--------------|----------|
| x86_64       | `RDI`    |
| AArch64      | `X0`     |
| RISC-V 64    | `A0`     |
| LoongArch 64 | `A0`     |

All other general-purpose registers are initialized to zero to prevent data leakage and ensure a clean environment.

## Stack Layout

The stack pointer (`SP`) is initialized to the top of the allocated stack (alignment requirements: 16-byte). 
The stack grows downwards. There is no initial data on the stack (no `argc`/`argv` yet).

## Argument Semantics (ARG0)

The `ARG0` value must follow these semantics:

| Value | Meaning |
|-------|---------|
| `0x0` | **None**. No startup context provided. |
| `0x600000` | **Boot Registry**. Only used by the initial process (Sprout). Pointer to the read-only module registry page. |
| `HandleId` | **Device Context**. Used for driver entry points. Represents the device the driver is intended to serve. |
| `Pointer` | **Startup Context**. Reserved for future use (e.g., `argc`/`argv` block). |

### Enforcement

Any task spawned with a value that does not match one of these defined semantics is considered invalid. The kernel shall enforce this at spawn time to prevent tasks from interpreting garbage as startup state.
