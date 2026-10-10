#![feature(rustc_attrs)]

#[rustc_dump_inferred_outlives]
enum Foo<'a, U> { //~ ERROR rustc_dump_inferred_outlives
    One(Bar<'a, U>)
}

struct Bar<'x, T> where T: 'x {
    x: &'x (),
    y: T,
}

fn main() {}
