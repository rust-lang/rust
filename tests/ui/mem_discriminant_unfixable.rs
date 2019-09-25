#![deny(clippy::mem_discriminant_non_enum)]

use std::mem;

enum Foo {
    One(usize),
    Two(u8),
}

struct A(Foo);

fn main() {
    // bad
    mem::discriminant(&"hello");
    mem::discriminant(&A(Foo::One(0)));
}
