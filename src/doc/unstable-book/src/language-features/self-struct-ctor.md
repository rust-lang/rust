# `self_struct_ctor`

The tracking issue for this feature is: [#51994]
[#51994]: https://github.com/rust-lang/rust/issues/51994

------------------------

The `self_struct_ctor` feature gate lets you use the special `Self`
identifier as a constructor and a pattern.

A simple example is:

```rust
#![feature(self_struct_ctor)]

struct ST(i32, i32);

impl ST {
    fn new() -> Self {
        ST(0, 1)
    }

    fn ctor() -> Self {
        Self(1,2)           // constructed by `Self`, it is the same as `ST(1, 2)`
    }

    fn pattern(self) {
        match self {
            Self(x, y) => println!("{} {}", x, y), // used as a pattern
        }
    }
}
```
