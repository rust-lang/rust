# Platform Layer Contract

This document defines the formal boundary between Thing-OS applications and the underlying platform, enforcing the principle that **stem is our std**.

## Overview

Thing-OS does not use Rust's standard library (`std`). Instead, we provide platform capabilities through the **stem** crate, which builds on `core` and `alloc` with our own platform abstraction layer (`stem::pal`).

### Why This Matters

- **Sustainability**: We can evolve platform APIs without forking rustc/Rust's std
- **Safety**: Enforced boundary prevents accidental std contamination
- **Clarity**: Platform capabilities are explicit and documented
- **Incremental**: New platform features are added intentionally, not accidentally

## The Platform Abstraction Layer (PAL)

The `stem::pal` module provides the **explicit contract** for all platform-specific functionality:

```rust
pub mod pal {
    pub mod log;      // Logging primitives
    pub mod clock;    // Time and monotonic clock
    pub mod abort;    // Panic and process termination
    pub mod alloc;    // Memory allocator hooks
}
```

## Panic/Unwind Policy (Thing-OS Targets)

Thing-OS targets are currently **abort-only** for panics.

- All Thing-OS target specs set `panic-strategy = "abort"`.
- Stack unwinding ABI/runtime (`panic_unwind`, personality routines, unwinder integration) is not part of the supported target contract.
- A panic in any thread aborts the process; panic payload propagation through `JoinHandle::join` is therefore not available on Thing-OS today.

To keep this failure mode explicit, Thing-OS runtime code emits a compile-time diagnostic if built with `panic = "unwind"`.

### Design Principles

1. **Explicit over implicit**: All platform capabilities must go through PAL
2. **Minimal and stable**: Only essential primitives are exposed
3. **Replaceable**: Implementations can be swapped without breaking consumers
4. **No std leakage**: Ensures `no_std` compliance throughout

## What's Allowed Where

### Kernel and Userspace (Runtime Code)

**MUST use:**
- `core` - Rust's core library (no allocator, no platform)
- `alloc` - Rust's allocator library (requires our global allocator)
- `stem` - Our platform layer (includes `stem::pal`)
- `abi` - Shared types and syscall interfaces

**MUST NOT use:**
- `std` - Rust's standard library
- Any crate that transitively depends on `std`

**How it's enforced:**
- `#![no_std]` attribute in every kernel/userspace crate
- `scripts/audit_platform_boundary.py` verifies compliance
- CI runs the audit on every commit

### Build Tools (Compile-time Code)

**CAN use std:**
- `xtask` - Build orchestration
- `tools/*` - Build-time utilities (pciids, bdd, unifont-gen, etc.)
- `*-macros` - Proc-macro crates (run at compile time)

These crates are clearly separated and never linked into the kernel or userspace binaries.

## Adding New Platform Capabilities

When you need a new platform capability (e.g., file I/O, networking), follow this process:

### 1. Define the PAL Interface

Add a new module under `stem/src/pal/`:

```rust
// stem/src/pal/fs.rs

/// Read bytes from a file.
pub fn read(fd: usize, buf: &mut [u8]) -> Result<usize, Errno> {
    crate::syscall::read(fd, buf)
}

/// Write bytes to a file.
pub fn write(fd: usize, buf: &[u8]) -> Result<usize, Errno> {
    crate::syscall::write(fd, buf)
}
```

### 2. Update PAL Module

Export the new module in `stem/src/pal/mod.rs`:

```rust
pub mod fs;  // New!
pub mod log;
pub mod clock;
// ...
```

### 3. Provide High-Level Wrappers

Add ergonomic wrappers in stem's public API:

```rust
// stem/src/fs.rs

pub use crate::pal::fs::{read, write};

pub struct File {
    fd: usize,
}

impl File {
    pub fn read(&mut self, buf: &mut [u8]) -> Result<usize, Errno> {
        pal::fs::read(self.fd, buf)
    }
}
```

### 4. Document the Contract

- Document the PAL function's semantics
- Explain syscall behavior and error conditions
- Note any platform-specific considerations

### 5. Test the Boundary

The platform boundary must remain auditable:

```bash
# Verify no std contamination
python3 scripts/audit_platform_boundary.py

# Run tests
cargo test -p stem
```

## Escape Hatches

### Host-Dependent Code

If you need host-specific functionality for development/testing, use the `xtask` crate or create a new crate under `tools/`. These are **never** linked into kernel/userspace.

Example:
```
tools/
  my-analyzer/     # Uses std, never linked into runtime
    Cargo.toml     # Does NOT have #![no_std]
    src/
      main.rs      # Uses std::fs, std::io, etc.
```

### Testing

Tests can use std via conditional compilation:

```rust
#![cfg_attr(not(test), no_std)]

#[cfg(test)]
mod tests {
    use std::vec::Vec;  // OK in tests
    // ...
}
```

## Compliance Verification

### Automated Audit

Run the platform boundary audit:

```bash
python3 scripts/audit_platform_boundary.py
```

This script:
- Verifies all kernel/userspace crates have `#![no_std]`
- Lists allowed std-using crates (build tools)
- Fails CI if violations are detected

### Manual Review

When reviewing code:
- ✅ New `use stem::pal::*` - Good, using the platform layer
- ✅ New `use core::*` or `use alloc::*` - Fine
- ❌ New `use std::*` in kernel/userspace - **REJECT**
- ❌ Missing `#![no_std]` in new userspace crate - **REJECT**

## Current Platform Surface

As of this writing, `stem::pal` provides:

| Module | Purpose | Key Functions |
|--------|---------|---------------|
| `pal::log` | Logging | `write()`, `write_with_provenance()` |
| `pal::clock` | Time | `monotonic_ns()`, `sleep_ns()`, `unix_time_ns()` |
| `pal::abort` | Termination | `abort()`, `debug_write_str()` |
| `pal::alloc` | Heap management | `grow_heap()` |

Higher-level APIs in stem build on these primitives:
- `stem::println!()` → `pal::log`
- `stem::sleep()` → `pal::clock`
- Panic handler → `pal::abort`
- Global allocator → `pal::alloc`

## Evolution Strategy

The PAL is intentionally minimal. Expand it **incrementally**:

1. Start with stubs or errors for unimplemented features
2. Add syscalls only when actually needed
3. Keep the PAL layer thin - business logic goes above it
4. Prefer composition over feature creep in PAL

Example progression:
- Phase 1: `pal::fs::read()` returns `Err(ENOSYS)`
- Phase 2: Implement basic read syscall
- Phase 3: Add buffering in `stem::io::BufReader` (not in PAL)

## FAQ

**Q: Can I use std in my userspace app?**  
A: No. Use `stem` instead. It provides logging, time, alloc, and more.

**Q: What if I need threading/async/sockets?**  
A: Add them to `stem::pal` first, then build ergonomic APIs on top in stem.

**Q: Can build tools use std?**  
A: Yes! `xtask` and crates under `tools/` can use std freely.

**Q: How do I know if my crate is compliant?**  
A: Run `python3 scripts/audit_platform_boundary.py`. It will tell you.

**Q: What if a dependency pulls in std?**  
A: Choose a `no_std` compatible alternative, or add `default-features = false` in Cargo.toml.

## References

- Source: `stem/src/pal/`
- Audit script: `scripts/audit_platform_boundary.py`
- CI integration: `.github/workflows/` (when added)
- Issue: [Original task for platform boundary formalization]
