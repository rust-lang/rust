# `fn_must_use`

The tracking issue for this feature is [#43302].

[#43302]: https://github.com/rust-lang/rust/issues/43302

------------------------

The `fn_must_use` feature allows functions and methods to be annotated with
`#[must_use]`, indicating that the `unused_must_use` lint should require their
return values to be used (similarly to how types annotated with `must_use`,
most notably `Result`, are linted if not used).

## Examples

```rust
#![feature(fn_must_use)]

#[must_use]
fn double(x: i32) -> i32 {
    2 * x
}

fn main() {
    double(4); // warning: unused return value of `double` which must be used

    let _ = double(4); // (no warning)
}

```
