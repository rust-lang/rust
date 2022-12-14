# `type_changing_struct_update`

The tracking issue for this feature is: [#86555]

[#86555]: https://github.com/rust-lang/rust/issues/86555

------------------------

This implements [RFC2528]. When turned on, you can create instances of the same struct
that have different generic type or lifetime parameters.

[RFC2528]: https://github.com/rust-lang/rfcs/blob/master/text/2528-type-changing-struct-update-syntax.md

```rust
#![allow(unused_variables, dead_code)]
#![feature(type_changing_struct_update)]

fn main () {
    struct Foo<T, U> {
        field1: T,
        field2: U,
    }

    let base: Foo<String, i32> = Foo {
        field1: String::from("hello"),
        field2: 1234,
    };
    let updated: Foo<f64, i32> = Foo {
        field1: 3.14,
        ..base
    };
}
```
