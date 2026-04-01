#![feature(rustc_attrs)]

#[rustc_dump_inferred_outlives]
struct Foo<'a, T: Iterator> { //~ ERROR rustc_dump_inferred_outlives
    bar: &'a T::Item
}

fn main() {}
