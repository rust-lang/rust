#![feature(rustc_attrs)]

#[rustc_dump_inferred_outlives]
struct Foo<'a, 'b, T> { //~ ERROR rustc_dump_inferred_outlives
    x: &'a &'b T
}

fn main() {}
