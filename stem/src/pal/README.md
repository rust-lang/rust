# Platform Abstraction Layer (PAL)

This module defines the **explicit contract** between Thing-OS applications and the underlying platform.

## Purpose

The PAL ensures that:
1. Platform capabilities are **explicit**, not implicit
2. The `no_std` boundary is enforced  
3. Platform code can evolve without breaking consumers
4. All platform-specific functionality goes through a single, auditable layer

## Modules

### `pal::log`
Logging primitives with level support (Error, Warn, Info, Debug, Trace).

**Use via stem macros:**
```rust
use stem::{error, warn, info, debug, trace};

info!("System initialized");
error!("Failed to read file: {}", err);
```

**Direct PAL usage (rare):**
```rust
use stem::pal::log::{Level, write};
write(Level::Info, format_args!("message"));
```

### `pal::clock`
Time and sleep primitives based on monotonic nanosecond counter.

**Use via stem functions:**
```rust
use stem::{sleep_ms, sleep, monotonic_ns, now};

sleep_ms(100);              // Sleep 100ms
let instant = now();        // Get current instant
let ns = monotonic_ns();    // Raw nanoseconds
```

**Direct PAL usage:**
```rust
use stem::pal::clock;
clock::sleep_ns(1_000_000);         // Sleep 1ms
let ns = clock::monotonic_ns();     // Get time
let unix_ns = clock::unix_time_ns(); // Unix timestamp
```

### `pal::abort`
Process termination and debug output for panic handlers.

**Use via stem (automatic):**
- Panic handler automatically uses `pal::abort`
- You rarely need to call this directly

**Direct PAL usage (very rare):**
```rust
use stem::pal::abort;
abort::debug_write_str("CRITICAL ERROR\n");
abort::abort(101);  // Terminate with exit code
```

### `pal::alloc`
Heap growth primitives for the global allocator.

**Use via stem's allocator (automatic):**
- Global allocator automatically calls `pal::alloc::grow_heap()`
- You never call this directly unless implementing a custom allocator

**Direct PAL usage (allocator internals only):**
```rust
use stem::pal::alloc;
alloc::grow_heap(256 * 1024)?;  // Grow heap by 256KB
```

## Design Principles

1. **Explicit > Implicit**: Every platform capability must be visible in PAL
2. **Minimal API**: Only essential primitives, no convenience wrappers here
3. **Syscall Boundary**: PAL wraps syscalls, higher-level APIs go in stem proper
4. **No std**: PAL is strictly `no_std`, depends only on `core` + syscalls

## Adding New Capabilities

See `docs/platform.md` for the complete guide to adding new platform features.

Quick summary:
1. Add new module under `stem/src/pal/`
2. Export it in `pal/mod.rs`
3. Provide ergonomic wrappers in stem's public API
4. Document the contract and test it

## Relationship to Stem

```
Userspace App
    ↓
stem (high-level APIs: println!, sleep(), File)
    ↓
stem::pal (platform primitives: log, clock, alloc)
    ↓
syscalls (kernel interface)
    ↓
Kernel
```

The PAL is the **platform contract layer** - it defines exactly what the platform provides.
