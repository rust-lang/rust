# `rvalue_static_promotion`

The tracking issue for this feature is: [#38865]

[#38865]: https://github.com/rust-lang/rust/issues/38865

------------------------

The `rvalue_static_promotion` feature allows directly creating `'static` references to
constant `rvalue`s, which in particular allowing for more concise code in the common case
in which a `'static` reference is all that's needed.


## Examples

```rust
#![feature(rvalue_static_promotion)]

fn main() {
    let DEFAULT_VALUE: &'static u32 = &42;
    assert_eq!(*DEFAULT_VALUE, 42);
}
```
