#![feature(precise_capturing)]

trait Foo {
    fn bar<'a>() -> impl Sized + use<Self>;
    //~^ ERROR `use<...>` precise capturing syntax is currently not allowed in return-position `impl Trait` in traits
}

fn main() {}
