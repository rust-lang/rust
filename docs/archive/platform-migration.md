# Platform Capability Migration Guide

This guide shows how to migrate existing platform-specific code into `stem::pal` or add new capabilities following the platform boundary contract.

## Quick Reference: Where Does Code Go?

```
┌─────────────────────────────────────────────────────┐
│  Userspace Applications                             │
│  - Can use: stem, abi, core, alloc                  │
│  - Must have: #![no_std]                            │
└─────────────────────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────┐
│  stem (High-Level APIs)                             │
│  - Ergonomic wrappers (File, Duration, etc.)       │
│  - Business logic                                    │
│  - Delegates to pal for platform operations         │
└─────────────────────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────┐
│  stem::pal (Platform Abstraction Layer)             │
│  - Thin syscall wrappers                            │
│  - Platform primitives only                         │
│  - Explicit contract                                │
└─────────────────────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────┐
│  Kernel (syscalls)                                  │
└─────────────────────────────────────────────────────┘
```

## Decision Tree: PAL vs Stem vs Userspace

**Is it a direct syscall wrapper?**
- YES → Goes in `stem::pal::*`
- NO → Continue...

**Does it require platform-specific knowledge (syscalls, hardware)?**
- YES → Goes in `stem::*` (delegates to pal)
- NO → Continue...

**Is it application-specific business logic?**
- YES → Goes in your userspace app
- NO → Consider if it belongs in a shared library

## Examples of Proper Placement

### Example 1: Time Operations

```
Platform Primitive (PAL):
  pal::clock::monotonic_ns() → syscall wrapper

High-Level API (stem):
  stem::now() → returns Instant (wraps pal)
  stem::sleep(duration) → nice API (wraps pal)
  
Userspace Usage:
  let start = stem::now();
  stem::sleep_ms(100);
  let elapsed = start.elapsed();
```

### Example 2: Logging

```
Platform Primitive (PAL):
  pal::log::write(Level, args) → syscall wrapper
  
High-Level API (stem):
  info!(...), error!(...) macros → nice API
  
Userspace Usage:
  stem::info!("System ready: version {}", VERSION);
```

### Example 3: Future - Networking (not yet implemented)

```
Platform Primitive (PAL):
  pal::net::socket_create() → syscall wrapper
  pal::net::socket_bind() → syscall wrapper
  pal::net::socket_send() → syscall wrapper
  
High-Level API (stem):
  stem::net::TcpListener → RAII wrapper
  stem::net::TcpStream → buffered I/O
  
Userspace Usage:
  let listener = TcpListener::bind("127.0.0.1:8080")?;
  let stream = listener.accept()?;
```

## Migration Checklist

When adding a new platform capability:

- [ ] 1. **Define the syscall** in `abi/src/syscall.rs`
- [ ] 2. **Implement in kernel** (`kernel/src/syscall/`)
- [ ] 3. **Add PAL wrapper** (`stem/src/pal/your_module.rs`)
  - Keep it thin (just syscall wrapper)
  - Document the contract
  - Use `Result<T, Errno>` for errors
- [ ] 4. **Export from PAL** (`stem/src/pal/mod.rs`)
- [ ] 5. **Add high-level API** (`stem/src/your_module.rs`)
  - Ergonomic types (RAII, builders, etc.)
  - Delegates to pal for platform operations
- [ ] 6. **Export from stem** (`stem/src/lib.rs`)
- [ ] 7. **Document** the API (doc comments)
- [ ] 8. **Test** in userspace
- [ ] 9. **Verify audit** (`python3 scripts/audit_platform_boundary.py`)

## Common Patterns

### Pattern 1: Simple Syscall Wrapper

```rust
// stem/src/pal/example.rs
use crate::syscall;
use abi::errors::Errno;

/// Get the current process ID.
#[inline]
pub fn getpid() -> u64 {
    syscall::getpid()
}
```

### Pattern 2: Error-Returning Syscall

```rust
// stem/src/pal/fs.rs
use crate::syscall;
use abi::errors::Errno;

/// Open a file and return a file descriptor.
pub fn open(path: &str, flags: u32) -> Result<usize, Errno> {
    syscall::fs_open(path, flags)
}
```

### Pattern 3: High-Level Wrapper with RAII

```rust
// stem/src/fs.rs
use crate::pal;
use abi::errors::Errno;

pub struct File {
    fd: usize,
}

impl File {
    pub fn open(path: &str) -> Result<Self, Errno> {
        let fd = pal::fs::open(path, 0)?;
        Ok(File { fd })
    }
}

impl Drop for File {
    fn drop(&mut self) {
        let _ = pal::fs::close(self.fd);
    }
}
```

## Anti-Patterns to Avoid

### ❌ Don't: Put Business Logic in PAL

```rust
// WRONG - stem/src/pal/json.rs
pub fn parse_json_config(path: &str) -> Result<Config, Error> {
    // Too high-level for PAL!
}
```

### ✅ Do: Keep PAL Thin, Logic in Stem

```rust
// RIGHT - stem/src/pal/fs.rs
pub fn read(fd: usize, buf: &mut [u8]) -> Result<usize, Errno> {
    syscall::fs_read(fd, buf)  // Just a syscall wrapper
}

// RIGHT - stem/src/config.rs
pub fn parse_json_config(path: &str) -> Result<Config, Error> {
    let mut file = File::open(path)?;  // Uses pal indirectly
    let content = file.read_to_string()?;
    serde_json::from_str(&content)
}
```

### ❌ Don't: Skip PAL and Call Syscalls Directly

```rust
// WRONG - userspace/my_app/src/main.rs
use abi::syscall::SYS_LOG_WRITE;

unsafe {
    syscall::syscall3(SYS_LOG_WRITE, ...);  // Bypasses PAL!
}
```

### ✅ Do: Always Go Through PAL/Stem

```rust
// RIGHT - userspace/my_app/src/main.rs
use stem::info;

info!("This goes through PAL properly");
```

## Validation

After adding a new capability, verify:

1. **Audit passes**:
   ```bash
   python3 scripts/audit_platform_boundary.py
   ```

2. **Documentation exists**:
   - PAL module has doc comments
   - High-level API has doc comments
   - Added to relevant README if needed

3. **Userspace can use it**:
   - Create a test app that uses the new capability
   - Verify it compiles with `#![no_std]`

4. **CI will catch violations**:
   - GitHub Actions workflow runs audit automatically
   - Build fails if std contamination detected

## Summary

The platform boundary ensures:
- ✅ Sustainable evolution (no std fork needed)
- ✅ Clear contracts (explicit PAL interface)
- ✅ Enforced boundaries (audit script)
- ✅ Incremental growth (add capabilities as needed)

Follow this guide to keep the codebase maintainable as the platform evolves.
