#![feature(rustc_attrs)]

#[rustc_outlives]
struct Foo<'a, T> { //~ ERROR rustc_outlives
    bar: &'a T,
}

fn main() {}
