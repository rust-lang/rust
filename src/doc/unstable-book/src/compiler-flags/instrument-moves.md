# `instrument-moves`

The `-Z instrument-moves` flag enables instrumentation of compiler-generated
move and copy operations, making them visible in profilers and stack traces
for performance debugging.

When enabled, the compiler will inject calls to `core::intrinsics::compiler_move`
and `core::intrinsics::compiler_copy` functions around large move and copy operations.
These functions are never actually executed (they contain `unreachable!()`), but
their presence in debug info makes expensive memory operations visible in profilers.

## Syntax

```bash
rustc -Z instrument-moves[=<boolean>]
rustc -Z instrument-moves-size-limit=<bytes>
```

## Options

- `-Z instrument-moves`: Enable/disable move/copy instrumentation (default: `false`)
- `-Z instrument-moves-size-limit=N`: Only instrument operations on types >= N bytes (default: 65 bytes)

## Examples

```bash
# Enable instrumentation with default threshold (pointer size)
rustc -Z instrument-moves main.rs

# Enable with custom 128-byte threshold
rustc -Z instrument-moves -Z instrument-moves-size-limit=128 main.rs

# Only instrument very large moves (1KB+)
rustc -Z instrument-moves -Z instrument-moves-size-limit=1024 main.rs
```

## Behavior

The instrumentation only applies to:
- Types larger than the specified size threshold
- Non-immediate types (those that would generate `memcpy`)
- Operations that actually move/copy data (not ZST types)

Stack traces will show the operations:
```text
0: memcpy
1: core::intrinsics::compiler_move::<MyLargeStruct, 148>
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
