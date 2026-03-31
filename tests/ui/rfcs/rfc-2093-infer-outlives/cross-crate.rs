#![feature(rustc_attrs)]

#[rustc_dump_inferred_outlives]
struct Foo<'a, T> { //~ ERROR rustc_dump_inferred_outlives
    bar: std::slice::IterMut<'a, T>
}

fn main() {}
