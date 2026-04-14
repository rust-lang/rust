# Syscalls v0.5 Contract

This document defines the authoritative ABI for ThingOS syscalls (v0.5).
All architectures and userspace libraries must adhere to this contract.

## stability

**Status**: Frozen for v0.5.
**Policy**: Syscall numbers and argument layouts are stable. New syscalls may be added, but existing ones cannot change without a revision bump.

## Calling Convention

### Common (All Arch)
- **Return**: `isize`
    - `>= 0`: Success (value depends on syscall)
    - `< 0`: Error (negated `Errno`, e.g. `-EINVAL`)
- **Arguments**: Up to 6 `usize` values.

### Architecture Specifics

| Arch | Syscall Number | Args (0..5) | Return | Trap |
|Data | Register | Registers | Register | Instruction |
|---|---|---|---|---|
| **x86_64** | `rax` | `rdi`, `rsi`, `rdx`, `r10`, `r8`, `r9` | `rax` | `syscall` |
| **AArch64** | `x8` | `x0`..`x5` | `x0` | `svc #0` |
| **RISC-V 64** | `a7` | `a0`..`a5` | `a0` | `ecall` |
| **LoongArch64** | `a7` | `a0`..`a5` | `a0` | `syscall 0` |

## Numbering Scheme

| Range | Usage |
|---|---|
| `0x0000` - `0x003F` | **Core** (Lifecycle, Debug, Basic IO) |
| `0x0040` - `0x007F` | **Time, Sched, Process** |
| `0x0080` - `0x00BF` | **Streams** (IO Channels) |
| `0x00C0` - `0x00FF` | **Watches / Events** |
| `0x0100` - `0x01FF` | **Root / Graph** (Reserved) |
| `0x0200`+ | Reserved |

## Global Syscall Table

### Core `0x00` - `0x3F`
| ID | Name | Args | Description |
|---|---|---|---|
| 1 | `SYS_EXIT` | `code: i32` | Terminate current task with exit code. |
| 2 | `SYS_DEBUG_WRITE` | `ptr: u64`, `len: u32` | Write UTF-8 string to debug output. |
| 3 | `SYS_SLEEP_MS` | `ms: u64` | Sleep for N milliseconds (or yield). |
| 4 | `SYS_DEVICE_CALL` | `call: *mut DeviceCall` | Invoke a device-specific operation. |

### Time/Sched `0x40` - `0x7F`
*Reserved for `SYS_TIME_NOW`, `SYS_YIELD`, `SYS_SPAWN`, etc.*

### Streams `0x80` - `0xBF`
| ID | Name | Args | Description |
|---|---|---|---|
| 128 | `SYS_STREAM_OPEN` | ... | Open a stream. |
| 129 | `SYS_STREAM_READ` | ... | Read from a stream. |
| 130 | `SYS_STREAM_POLL` | ... | Poll stream status. |

### Watches `0xC0` - `0xFF`
| ID | Name | Args | Description |
|---|---|---|---|
| 192 | `SYS_WATCH_SUBSCRIBE` | ... | Subscribe to graph changes. |
| 193 | `SYS_WATCH_READ` | ... | Read event queue. |

## Validation Rules
1. **User Pointers**: Must be validated against user address space range before access.
2. **Alignment**: Pointers must be naturally aligned for the type they point to.
3. **Strings**: `DEBUG_WRITE` expects valid UTF-8. Partial/invalid UTF-8 may be rejected or replaced.
