#![feature(rustc_attrs)]

#[rustc_dump_inferred_outlives]
enum Foo<'a, T> { //~ ERROR rustc_dump_inferred_outlives

    One(Bar<'a, T>)
}

struct Bar<'b, U> {
    field2: &'b U
}

fn main() {}
