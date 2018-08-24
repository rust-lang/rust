#![feature(rustc_attrs)]
#![feature(infer_outlives_requirements)]

#[rustc_outlives]
enum Foo<'a, U> { //~ ERROR 15:1: 17:2: rustc_outlives
    One(Bar<'a, U>)
}

struct Bar<'x, T> where T: 'x {
    x: &'x (),
    y: T,
}

fn main() {}

