# `annotate-moves`

The `-Z annotate-moves` flag enables annotation of compiler-generated
move and copy operations, making them visible in profilers and stack traces
for performance debugging.

When enabled, the compiler will inject calls to `core::profiling::compiler_move`
and `core::profiling::compiler_copy` functions around large move and copy operations.
These functions are never actually executed (they contain `unreachable!()`), but
their presence in debug info makes expensive memory operations visible in profilers.

## Syntax

```bash
rustc -Z annotate-moves[=<value>]
```

Where `<value>` can be:
- A boolean: `true`, `false`, `yes`, `no`, `on`, `off`
- A number: size threshold in bytes (e.g., `128`)
- Omitted: enables with default threshold (65 bytes)

## Options

- `-Z annotate-moves` or `-Z annotate-moves=true`: Enable with default size limit (65 bytes)
- `-Z annotate-moves=false`: Disable annotation
- `-Z annotate-moves=N`: Enable with custom size limit of N bytes

## Examples

```bash
# Enable annotation with default threshold (65 bytes)
rustc -Z annotate-moves main.rs

# Enable with custom 128-byte threshold
rustc -Z annotate-moves=128 main.rs

# Only annotate very large moves (1KB+)
rustc -Z annotate-moves=1024 main.rs

# Explicitly disable
rustc -Z annotate-moves=false main.rs
```

## Behavior

The annotation only applies to:
- Types larger than the specified size threshold
- Non-immediate types (those that would generate `memcpy`)
- Operations that actually move/copy data (not ZST types)

Stack traces will show the operations:
```text
0: memcpy
1: core::profiling::compiler_move::<MyLargeStruct, 148>
2: my_function
```

## Example

```rust
#[derive(Clone)]
struct LargeData {
    buffer: [u8; 1000],
}

fn example() {
    let data = LargeData { buffer: [0; 1000] };
    let copy = data.clone(); // Shows as compiler_copy in profiler
    let moved = data;        // Shows as compiler_move in profiler
}
```

## Overhead

This has no effect on generated code; it only adds debuginfo. The overhead is
typically very small; on rustc itself, the default limit of 65 bytes adds about
0.055% to the binary size.
