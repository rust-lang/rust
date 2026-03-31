#![feature(rustc_attrs)]

trait Trait<'x, T> where T: 'x {
}

#[rustc_dump_inferred_outlives]
struct Foo<'a, A> //~ ERROR rustc_dump_inferred_outlives
{
    foo: Box<dyn Trait<'a, A>>
}

fn main() {}
