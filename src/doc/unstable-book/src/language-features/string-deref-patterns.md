# `string_deref_patterns`

The tracking issue for this feature is: [#87121]

[#87121]: https://github.com/rust-lang/rust/issues/87121

------------------------

> **Note**: This feature will be superseded by [`deref_patterns`] in the future.

This feature permits pattern matching `String` to `&str` through [its `Deref` implementation].

```rust
#![feature(string_deref_patterns)]

pub enum Value {
    String(String),
    Number(u32),
}

pub fn is_it_the_answer(value: Value) -> bool {
    match value {
        Value::String("42") => true,
        Value::Number(42) => true,
        _ => false,
    }
}
```

Without this feature other constructs such as match guards have to be used.

```rust
# pub enum Value {
#    String(String),
#    Number(u32),
# }
#
pub fn is_it_the_answer(value: Value) -> bool {
    match value {
        Value::String(s) if s == "42" => true,
        Value::Number(42) => true,
        _ => false,
    }
}
```

[`deref_patterns`]: ./deref-patterns.md
[its `Deref` implementation]: https://doc.rust-lang.org/std/string/struct.String.html#impl-Deref-for-String
