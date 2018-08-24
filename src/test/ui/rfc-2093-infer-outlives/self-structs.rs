#![feature(rustc_attrs)]
#![feature(infer_outlives_requirements)]

#[rustc_outlives]
struct Foo<'a, 'b, T> { //~ ERROR 15:1: 17:2: rustc_outlives
    field1: Bar<'a, 'b, T>
}

trait Bar<'x, 's, U>
    where U: 'x,
    Self:'s
{}

fn main() {}
