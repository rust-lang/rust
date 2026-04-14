# Platform Abstraction Layer (PAL) Examples

This document shows practical examples of how to use `stem::pal` correctly.

## Example 1: Adding Custom Logging

Most code should use the high-level macros, but if you need direct control:

```rust
#![no_std]
extern crate alloc;
use stem::pal::log::{Level, write};

pub fn custom_log_system_event(event: &str) {
    // Direct PAL usage for special logging needs
    write(
        Level::Info, 
        format_args!("[SYSTEM] {}", event)
    );
}
```

**Better approach** - use stem's macros:

```rust
#![no_std]
use stem::info;

pub fn log_system_event(event: &str) {
    info!("[SYSTEM] {}", event);
}
```

## Example 2: Measuring Elapsed Time

Using PAL clock primitives for precise timing:

```rust
#![no_std]
use stem::pal::clock;

pub fn measure_operation<F: FnOnce()>(f: F) -> u64 {
    let start = clock::monotonic_ns();
    f();
    let end = clock::monotonic_ns();
    end - start  // Returns elapsed nanoseconds
}
```

**Better approach** - use stem's Instant type:

```rust
#![no_std]
use stem::{now, Duration};

pub fn measure_operation<F: FnOnce()>(f: F) -> Duration {
    let start = now();
    f();
    start.elapsed()
}
```

## Example 3: Adding a New Platform Capability (File I/O)

Here's how you would add file I/O to the platform (hypothetical):

### Step 1: Add PAL Module

Create `stem/src/pal/fs.rs`:

```rust
//! File system platform abstraction.

use crate::syscall;
use abi::errors::Errno;

/// Open a file and return a file descriptor.
pub fn open(path: &str, flags: u32) -> Result<usize, Errno> {
    syscall::fs_open(path, flags)
}

/// Read bytes from a file descriptor.
pub fn read(fd: usize, buf: &mut [u8]) -> Result<usize, Errno> {
    syscall::fs_read(fd, buf)
}

/// Write bytes to a file descriptor.
pub fn write(fd: usize, buf: &[u8]) -> Result<usize, Errno> {
    syscall::fs_write(fd, buf)
}

/// Close a file descriptor.
pub fn close(fd: usize) -> Result<(), Errno> {
    syscall::fs_close(fd)
}
```

### Step 2: Export in PAL

Update `stem/src/pal/mod.rs`:

```rust
pub mod log;
pub mod clock;
pub mod abort;
pub mod alloc;
pub mod fs;  // New!
```

### Step 3: Add High-Level Wrapper

Create `stem/src/fs.rs`:

```rust
//! File system operations.

use crate::pal;
use abi::errors::Errno;
use alloc::string::String;
use alloc::vec::Vec;

pub struct File {
    fd: usize,
}

impl File {
    pub fn open(path: &str) -> Result<Self, Errno> {
        let fd = pal::fs::open(path, 0)?;
        Ok(File { fd })
    }
    
    pub fn read(&mut self, buf: &mut [u8]) -> Result<usize, Errno> {
        pal::fs::read(self.fd, buf)
    }
    
    pub fn read_to_string(&mut self) -> Result<String, Errno> {
        let mut buf = Vec::new();
        let mut chunk = [0u8; 4096];
        loop {
            let n = self.read(&mut chunk)?;
            if n == 0 { break; }
            buf.extend_from_slice(&chunk[..n]);
        }
        String::from_utf8(buf).map_err(|_| Errno::EINVAL)
    }
}

impl Drop for File {
    fn drop(&mut self) {
        let _ = pal::fs::close(self.fd);
    }
}
```

### Step 4: Use It

```rust
#![no_std]
extern crate alloc;
use stem::fs::File;

fn read_config() -> Result<(), stem::abi::errors::Errno> {
    let mut f = File::open("/etc/config.txt")?;
    let content = f.read_to_string()?;
    stem::info!("Config: {}", content);
    Ok(())
}
```

## Example 4: Platform-Independent Userspace App

A complete userspace application using only PAL primitives:

```rust
#![no_std]
#![no_main]

extern crate alloc;
extern crate stem;

use stem::{info, sleep_ms, monotonic_ns};
use alloc::format;

#[stem::main]
fn main() {
    info!("Application starting...");
    
    let start = monotonic_ns();
    
    // Do work
    for i in 0..10 {
        info!("Iteration {}", i);
        sleep_ms(100);
    }
    
    let elapsed = monotonic_ns() - start;
    info!("Completed in {}ms", elapsed / 1_000_000);
}
```

This app is **completely platform-independent** because:
- Uses `#![no_std]` - no std library
- Uses `extern crate stem` - our platform layer
- All platform operations go through `stem::pal`
- Will work on x86_64, aarch64, riscv64, etc.

## Anti-Patterns (DON'T DO THIS)

### ❌ Using std

```rust
// WRONG - This will fail the platform audit
use std::time::Instant;  // std::time is not available!

fn bad_timing() {
    let now = Instant::now();  // Compile error in no_std
}
```

### ❌ Bypassing PAL

```rust
// WRONG - Direct syscall without going through PAL
use abi::syscall::SYS_LOG_WRITE;

fn bad_log(msg: &str) {
    unsafe {
        syscall::syscall3(SYS_LOG_WRITE, ...);  // Fragile, skips PAL
    }
}
```

**Right way:**
```rust
use stem::pal::log::{Level, write};

fn good_log(msg: &str) {
    write(Level::Info, format_args!("{}", msg));
}
```

### ❌ Feature Creep in PAL

```rust
// WRONG - Too high-level for PAL
pub mod pal::json {
    pub fn parse(s: &str) -> JsonValue { ... }  // Business logic!
}
```

PAL should be **thin primitives only**. High-level functionality goes in stem proper:

```rust
// RIGHT - Keep PAL thin
pub mod pal::fs {
    pub fn read(fd: usize, buf: &[u8]) -> Result<usize, Errno> { ... }
}

// High-level wrapper in stem
pub mod json {
    pub fn parse_from_file(path: &str) -> Result<JsonValue, Error> {
        let mut f = fs::File::open(path)?;
        let s = f.read_to_string()?;
        // JSON parsing here
    }
}
```

## Key Takeaways

1. **Use stem's high-level APIs** (macros, functions) for most code
2. **PAL is for platform primitives** (syscall wrappers, thin abstractions)
3. **Keep PAL minimal** - high-level features go in stem proper
4. **Never use std** in kernel/userspace - always `#![no_std]`
5. **Run the audit** regularly: `python3 scripts/audit_platform_boundary.py`

## See Also

- `docs/platform.md` - Full platform contract
- `stem/src/pal/README.md` - PAL module overview
- `AGENTS.md` - Quick reference for the platform boundary
