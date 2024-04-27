#![feature(rustc_attrs)]

#[rustc_outlives]
struct Foo<'a, T: Iterator> { //~ ERROR rustc_outlives
    bar: &'a T::Item
}

fn main() {}
