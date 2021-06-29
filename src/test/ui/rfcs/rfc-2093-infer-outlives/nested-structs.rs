#![feature(rustc_attrs)]

#[rustc_outlives]
struct Foo<'a, T> { //~ ERROR rustc_outlives
    field1: Bar<'a, T>
}

struct Bar<'b, U> {
    field2: &'b U
}

fn main() {}
