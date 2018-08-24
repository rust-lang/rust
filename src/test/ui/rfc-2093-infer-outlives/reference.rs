#![feature(rustc_attrs)]
#![feature(infer_outlives_requirements)]

#[rustc_outlives]
struct Foo<'a, T> { //~ ERROR rustc_outlives
    bar: &'a T,
}

fn main() {}
