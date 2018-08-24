#![feature(rustc_attrs)]
#![feature(infer_outlives_requirements)]


#[rustc_outlives]
enum Foo<'a, T> { //~ ERROR 16:1: 19:2: rustc_outlives

    One(Bar<'a, T>)
}

struct Bar<'b, U> {
    field2: &'b U
}

fn main() {}
