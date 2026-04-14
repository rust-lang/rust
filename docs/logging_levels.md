# ThingOS Logging Level Policy

This document defines the canonical logging level semantics for ThingOS kernel and userspace code.

## Quick Reference

| Level   | When to Use | Examples | Noise Level |
|---------|-------------|----------|-------------|
| `error!` | Operation failed, correctness impacted | "failed to map region", "device init failed", "corrupt data" | Rare |
| `warn!` | Unexpected but recoverable | "falling back to safe mode", "timeout, retrying", "unexpected device response" | Occasional |
| `info!` | High-level lifecycle milestones | "bloom started", "virtio-gpu initialized", "network configured" | Sparse |
| `debug!` | Developer diagnostics | "allocated framebuffer pool (n=3)", "state transition Free → Acquired" | Moderate |
| `trace!` | Extremely verbose, per-event/per-iteration | per-drawlist op, per-interrupt, per-packet, per-node traversal | Very High |

## Detailed Semantics

### `error!` - Correctness Impact

**Use when:** An operation failed and correctness is impacted. The system state is inconsistent or the requested operation cannot be completed.

**Guidelines:**
- The error represents a real failure, not an expected condition
- Include actionable context: what failed, why, what happens next
- Include key identifiers: device ID, node ID, process ID, error code
- Should be rare in normal operation

**Examples:**
```rust
kerror!("failed to map region {:#x}-{:#x}: {:?}", start, end, err);
error!("device {} init failed; continuing without it", dev_id);
kerror!("corrupt data in bytespace {}: invariant broken", bs_id);
error!("panic in task {}: {}", tid, msg);
```

**Anti-patterns:**
```rust
// Don't use error for expected failures
error!("file not found"); // Use warn! or debug! instead

// Don't use error for recoverable conditions
error!("retrying connection"); // Use warn! instead
```

### `warn!` - Recoverable Unexpected Condition

**Use when:** Something unexpected happened, but the system can continue safely. The operation may have degraded performance or functionality.

**Guidelines:**
- The condition is unexpected but handled
- The system adapts or retries
- May indicate configuration issues or environmental problems
- Should not spam (consider rate limiting for recurring warnings)

**Examples:**
```rust
warn!("falling back to software rendering");
kwarn!("timeout waiting for device {}; retrying", dev_id);
warn!("unexpected device response: {:02x}; ignoring", resp);
warn!("feature {} not supported; using fallback", feat);
```

**Anti-patterns:**
```rust
// Don't warn about normal operation
warn!("processing packet"); // Use trace! or debug! instead

// Don't warn about things that happen frequently
warn!("buffer full"); // If this is common, use debug! or implement rate limiting
```

### `info!` - User-Visible Milestones

**Use when:** A high-level lifecycle event or user-meaningful milestone occurs. This is the "story level" of system operation.

**Guidelines:**
- Should be sparse and stable (not chatty)
- Represents completed milestones, not in-progress work
- Should make sense to a user reading the console
- Typically one-time or very infrequent events
- Boot sequence milestones, service startup, device initialization

**Examples:**
```rust
info!("bloom started");
kinfo!("virtio-gpu initialized: {}x{}", width, height);
info!("network configured: {}", ip);
info!("loaded module: {}", name);
kinfo!("scheduler running");
```

**Anti-patterns:**
```rust
// Don't log repetitive events
info!("rendering frame {}", frame_num); // Use debug! or trace! instead

// Don't log intermediate steps
info!("allocating buffer..."); // Use debug! instead

// Don't log per-operation details
info!("processing request {}", req_id); // Use debug! or trace! instead
```

### `debug!` - Developer Diagnostics

**Use when:** Information is useful for developers diagnosing subsystem behavior, but not needed during normal operation.

**Guidelines:**
- Moderate noise is acceptable
- Avoid per-frame, per-irq, per-packet spam
- State transitions are good candidates
- Configuration and capability details
- Resource allocation summaries (not per-allocation)

**Examples:**
```rust
debug!("allocated framebuffer pool: {} buffers", count);
kdebug!("received feature bits: {:032b}", bits);
debug!("state transition: {:?} → {:?}", old, new);
debug!("device capabilities: {:?}", caps);
kdebug!("mapped region: {:#x}-{:#x}", start, end);
```

**Anti-patterns:**
```rust
// Don't log in tight loops
for op in drawlist {
    debug!("processing op: {:?}", op); // Use trace! instead
}

// Don't log every allocation
debug!("allocated {} bytes", size); // Too noisy, use trace! or remove
```

### `trace!` - Extremely Verbose

**Use when:** You need to see every single event, iteration, or packet. This is for deep debugging and should be very noisy.

**Guidelines:**
- Anything that might appear in tight loops
- Per-event, per-iteration, per-packet, per-interrupt, per-frame logging
- Assume this level is disabled by default
- Performance impact is acceptable when enabled
- Use for understanding exact execution flow

**Examples:**
```rust
trace!("drawlist op: {:?}", op);
ktrace!("interrupt: vector {}, cpu {}", vec, cpu);
trace!("packet received: {} bytes", len);
trace!("traversing node: {}", node_id);
ktrace!("task switch: {} → {}", old_tid, new_tid);
```

**Anti-patterns:**
```rust
// Don't use trace for one-time events
trace!("system initialized"); // Use info! or debug! instead
```

## Special Cases

### Kernel-Specific Macros

The kernel uses `k`-prefixed variants:
- `kerror!`, `kwarn!`, `kinfo!`, `kdebug!`, `ktrace!`

There are also special kernel macros:
- `contract!` - Always logged, regardless of level (critical boot milestones)
- `log_event!` - Structured logging with key-value fields
- `kprint!`, `kprintln!` - Raw output (use sparingly, prefer level-specific macros)

### Boot Sequence

During boot, prefer `contract!` for critical milestones that must always be visible:
```rust
contract!("Entering kernel");
contract!("Memory initialized");
contract!("Scheduler started");
```

Use `kinfo!` for normal boot milestones, `kdebug!` for details.

### Hot Paths

**Hot paths** are code that executes frequently (per-frame, per-interrupt, per-packet, etc.).

**Rules for hot paths:**
1. Default to `trace!` for any per-event logging
2. No logging at `info!` level in hot paths
3. No expensive formatting (`format!()`, large `{:?}` dumps) in `debug!` or `trace!` unless guarded
4. Consider removing logging entirely if not needed

**Hot path examples:**
- Compositor render loop
- Interrupt handlers
- Packet processing
- Memory allocator
- Scheduler tick/task switch
- Graph traversal
- Present queue operations

### Performance Guidelines

1. **Avoid allocation in disabled logs:**
   ```rust
   // BAD: allocates even when debug is disabled
   kdebug!("data: {}", format_args!(...));
   
   // GOOD: logging macros defer formatting
   kdebug!("data: {}", value);
   ```

2. **Guard expensive formatting:**
   ```rust
   // If you must do expensive work:
   if log::log_enabled!(log::Level::Debug) {
       let expensive_dump = compute_dump();
       kdebug!("dump: {}", expensive_dump);
   }
   ```

3. **Use structured logging for complex data:**
   ```rust
   // Instead of formatting a large struct:
   log_event!("device_init", {
       "id" => device_id,
       "vendor" => vendor,
       "status" => status,
   });
   ```

## Mechanical Rules (Quick Decisions)

When in doubt, apply these heuristics:

### By Message Content

| Message Contains | Default Level | Rationale |
|------------------|---------------|-----------|
| "enter", "exit", "tick", "loop" | `trace!` | Per-iteration |
| "received", "render op", "irq", "packet" | `trace!` | Per-event |
| "allocated", "freed", "mapped" | `debug!` | Per-operation detail |
| "initialized", "started", "loaded", "ready" | `info!` | Lifecycle milestone |
| "state transition", "feature bits", "capability" | `debug!` | Developer diagnostic |
| "fallback", "retry", "timeout", "unexpected" | `warn!` | Recoverable issue |
| "failed", "panic", "corrupt", "invariant" | `error!` | Correctness failure |

### By Frequency

| How Often | Level |
|-----------|-------|
| Once per boot | `info!` or `debug!` |
| Once per device/service | `info!` |
| Once per operation | `debug!` |
| Per frame/tick/interrupt | `trace!` |
| Per packet/event | `trace!` |

### By Audience

| Who Cares | Level |
|-----------|-------|
| End user | `info!`, `warn!`, `error!` |
| System administrator | `info!`, `warn!`, `error!` |
| Developer debugging this subsystem | `debug!` |
| Developer debugging this specific function | `trace!` |

## Migration Checklist

When auditing a module:

1. [ ] Identify hot paths (loops, interrupt handlers, per-event code)
2. [ ] Ensure hot paths use `trace!` or no logging
3. [ ] Ensure lifecycle events use `info!`
4. [ ] Ensure errors use `error!`, warnings use `warn!`
5. [ ] Check for expensive formatting in `debug!`/`trace!`
6. [ ] Verify no `println!` or `dbg!` in production code
7. [ ] Test with default log level (should be readable, not spammy)
8. [ ] Test with debug level enabled (should provide useful detail)
9. [ ] Test with trace level enabled (should be very verbose)

## Banned Patterns

### ❌ `println!` in kernel/userspace

```rust
// BAD: bypasses logging system
println!("Processing request");

// GOOD: use appropriate level
info!("Processing request");
```

**Exception:** Early boot before logging is initialized, or build tools (xtask).

### ❌ `dbg!` anywhere

```rust
// BAD: debug-only, unpredictable output
dbg!(some_value);

// GOOD: use appropriate level with context
debug!("value: {:?}", some_value);
```

### ❌ Expensive formatting without guards

```rust
// BAD: allocates even when trace is disabled
trace!("data: {:?}", expensive_clone());

// GOOD: guard expensive work
if log_enabled!(Level::Trace) {
    trace!("data: {:?}", expensive_clone());
}
```

## Rate Limiting (Future)

For warnings that may occur frequently, consider adding rate-limiting utilities:

```rust
// Warn only once per unique key
warn_once!("missing_feature", "Feature X not supported; using fallback");

// Warn at most every N seconds
warn_every!(Duration::from_secs(30), "high_latency", "Latency spike: {}ms", lat);
```

**Note:** These utilities are not yet implemented. For now, consider:
- Moving recurring warnings to `debug!`
- Using a manual guard with a static flag for "warn once" behavior

## Testing

### Default Configuration

```bash
just build && just run
```

**Expected:** Console shows lifecycle milestones (info/warn/error) without spam. Should read like a story of system startup.

### Debug Configuration

Set minimum log level to debug (implementation-specific).

**Expected:** Additional diagnostic detail appears, but still manageable volume.

### Trace Configuration  

Set minimum log level to trace.

**Expected:** Very verbose output, including per-event details. May impact performance.

## Summary

- **Keep `info!` sparse:** Only lifecycle milestones
- **Use `warn!` for recoverable issues:** Not normal operation
- **Use `error!` for real failures:** Correctness impact
- **Push details to `debug!`:** Developer diagnostics
- **Push hot-path spam to `trace!`:** Per-event/per-iteration
- **No `println!` or `dbg!` in production code**
- **Guard expensive formatting in `debug!`/`trace!`**

When in doubt, **prefer a lower level** (more verbose). It's easier to elevate an important message than to filter out spam.
