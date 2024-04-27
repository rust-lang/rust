#![feature(rustc_attrs)]

#[rustc_outlives]
struct Foo<'a, 'b, T> { //~ ERROR rustc_outlives
    x: &'a &'b T
}

fn main() {}
