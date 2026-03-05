#![feature(rustc_attrs)]

#[rustc_dump_inferred_outlives]
struct Foo<'a, 'b, T> { //~ ERROR rustc_dump_inferred_outlives
    field1: dyn Bar<'a, 'b, T>
}

trait Bar<'x, 's, U>
    where U: 'x,
    Self:'s
{}

fn main() {}
