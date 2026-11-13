# `adt_const_params`

The tracking issue for this feature is: [#95174]

[#95174]: https://github.com/rust-lang/rust/issues/95174

------------------------

Allows for using more complex types for const parameters, such as structs or enums.

```rust
#![feature(adt_const_params)]
#![allow(incomplete_features)]

use std::marker::ConstParamTy;

#[derive(ConstParamTy, PartialEq, Eq)]
enum Foo {
    A,
    B,
    C,
}

#[derive(ConstParamTy, PartialEq, Eq)]
struct Bar {
    flag: bool,
}

fn is_foo_a_and_bar_true<const F: Foo, const B: Bar>() -> bool {
    match (F, B.flag) {
        (Foo::A, true) => true,
        _ => false,
    }
}
```
