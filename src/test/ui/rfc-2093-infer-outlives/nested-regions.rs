#![feature(rustc_attrs)]
#![feature(infer_outlives_requirements)]

#[rustc_outlives]
struct Foo<'a, 'b, T> { //~ ERROR 15:1: 17:2: rustc_outlives
    x: &'a &'b T
}

fn main() {}
