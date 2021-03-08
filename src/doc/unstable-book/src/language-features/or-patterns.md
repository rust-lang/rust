# `or_patterns`

The tracking issue for this feature is: [#54883]

[#54883]: https://github.com/rust-lang/rust/issues/54883

------------------------

The `or_pattern` language feature allows `|` to be arbitrarily nested within
a pattern, for example, `Some(A(0) | B(1 | 2))` becomes a valid pattern.

## Examples

```rust,no_run
#![feature(or_patterns)]

pub enum Foo {
    Bar,
    Baz,
    Quux,
}

pub fn example(maybe_foo: Option<Foo>) {
    match maybe_foo {
        Some(Foo::Bar | Foo::Baz) => {
            println!("The value contained `Bar` or `Baz`");
        }
        Some(_) => {
            println!("The value did not contain `Bar` or `Baz`");
        }
        None => {
            println!("The value was `None`");
        }
    }
}
```
