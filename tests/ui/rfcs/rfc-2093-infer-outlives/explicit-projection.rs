#![feature(rustc_attrs)]

trait Trait<'x, T> where T: 'x {
    type Type;
}

#[rustc_outlives]
struct Foo<'a, A, B> where A: Trait<'a, B> //~ ERROR rustc_outlives
{
    foo: <A as Trait<'a, B>>::Type
}

fn main() {}
