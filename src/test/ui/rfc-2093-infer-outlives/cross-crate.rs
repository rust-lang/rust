#![feature(rustc_attrs)]
#![feature(infer_outlives_requirements)]

#[rustc_outlives]
struct Foo<'a, T> { //~ ERROR 15:1: 17:2: rustc_outlives
    bar: std::slice::IterMut<'a, T>
}

fn main() {}

