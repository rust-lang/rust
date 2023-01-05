#![feature(rustc_attrs)]

#[rustc_outlives]
struct Foo<'a, T> { //~ ERROR rustc_outlives
    bar: std::slice::IterMut<'a, T>
}

fn main() {}
