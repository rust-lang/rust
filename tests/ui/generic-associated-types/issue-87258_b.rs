#![feature(type_alias_impl_trait)]

// See https://github.com/rust-lang/rust/issues/87258#issuecomment-883293367

trait Trait1 {}

struct Struct<'b>(&'b ());

impl<'d> Trait1 for Struct<'d> {}

pub trait Trait2 {
    type FooFuture<'a>: Trait1;
    fn foo<'a>() -> Self::FooFuture<'a>;
}

type Helper<'xenon, 'yttrium, KABOOM: Trait2> = impl Trait1;
//~^ ERROR unconstrained opaque type

impl<'c, S: Trait2> Trait2 for &'c mut S {
    type FooFuture<'a> = Helper<'c, 'a, S>;
    fn foo<'a>() -> Self::FooFuture<'a> {
        Struct(unimplemented!())
    }
}

fn main() {}
