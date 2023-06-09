# `unboxed_closures`

The tracking issue for this feature is [#29625]

See Also: [`fn_traits`](../library-features/fn-traits.md)

[#29625]: https://github.com/rust-lang/rust/issues/29625

----

The `unboxed_closures` feature allows you to write functions using the `"rust-call"` ABI,
required for implementing the [`Fn*`] family of traits. `"rust-call"` functions must have
exactly one (non self) argument, a tuple representing the argument list.

[`Fn*`]: ../../std/ops/trait.Fn.html

```rust
#![feature(unboxed_closures)]

extern "rust-call" fn add_args(args: (u32, u32)) -> u32 {
    args.0 + args.1
}

fn main() {}
```
