# `tool_attributes`

The tracking issue for this feature is: [#44690]

[#44690]: https://github.com/rust-lang/rust/issues/44690

------------------------

Tool attributes let you use scoped attributes to control the behavior
of certain tools.

Currently tool names which can be appear in scoped attributes are restricted to
`clippy` and `rustfmt`.

## An example

```rust
#![feature(tool_attributes)]

#[rustfmt::skip]
fn foo() { println!("hello, world"); }

fn main() {
    foo();
}
```
