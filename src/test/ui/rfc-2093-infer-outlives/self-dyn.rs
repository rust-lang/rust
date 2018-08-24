#![feature(dyn_trait)]
#![feature(rustc_attrs)]
#![feature(infer_outlives_requirements)]

trait Trait<'x, 's, T> where T: 'x,
      's: {
}

#[rustc_outlives]
struct Foo<'a, 'b, A> //~ ERROR 20:1: 23:2: rustc_outlives
{
    foo: Box<dyn Trait<'a, 'b, A>>
}

fn main() {}
