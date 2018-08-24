#![feature(dyn_trait)]
#![feature(rustc_attrs)]
#![feature(infer_outlives_requirements)]

trait Trait<'x, T> where T: 'x {
}

#[rustc_outlives]
struct Foo<'a, A> //~ ERROR 19:1: 22:2: rustc_outlives
{
    foo: Box<dyn Trait<'a, A>>
}

fn main() {}
