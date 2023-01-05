#![feature(rustc_attrs)]

#[rustc_outlives]
struct Foo<'b, U> { //~ ERROR rustc_outlives
    bar: Bar<'b, U>
}

struct Bar<'a, T> where T: 'a {
    x: &'a (),
    y: T,
}

fn main() {}
