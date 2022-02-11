#![feature(type_alias_impl_trait)]
#![feature(generic_associated_types)]

// See https://github.com/rust-lang/rust/issues/87258#issuecomment-883293367

trait Trait1 {}

struct Struct<'b>(&'b ());

impl<'d> Trait1 for Struct<'d> {}

pub trait Trait2 {
    type FooFuture<'a>: Trait1;
    fn foo<'a>() -> Self::FooFuture<'a>;
}

impl<'c, S: Trait2> Trait2 for &'c mut S {
    type FooFuture<'a> = impl Trait1;
    fn foo<'a>() -> Self::FooFuture<'a> { //~ ERROR
        Struct(unimplemented!())
    }
}

fn main() {}
