# `c_variadic`

The tracking issue for this feature is: [#44930]

[#44930]: https://github.com/rust-lang/rust/issues/44930

------------------------

The `c_variadic` language feature enables C-variadic functions to be
defined in Rust. They may be called both from within Rust and via FFI.

## Examples

```rust
#![feature(c_variadic)]

pub unsafe extern "C" fn add(n: usize, mut args: ...) -> usize {
    let mut sum = 0;
    for _ in 0..n {
        sum += args.arg::<usize>();
    }
    sum
}
```
