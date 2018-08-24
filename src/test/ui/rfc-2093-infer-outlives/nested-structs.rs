#![feature(rustc_attrs)]
#![feature(infer_outlives_requirements)]

#[rustc_outlives]
struct Foo<'a, T> { //~ ERROR 15:1: 17:2: rustc_outlives
    field1: Bar<'a, T>
}

struct Bar<'b, U> {
    field2: &'b U
}

fn main() {}
