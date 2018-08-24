#![feature(rustc_attrs)]
#![feature(infer_outlives_requirements)]

#[rustc_outlives]
struct Foo<'a, T: Iterator> { //~ ERROR rustc_outlives
    bar: &'a T::Item
}

fn main() {}

