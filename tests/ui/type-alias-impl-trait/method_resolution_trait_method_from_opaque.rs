//! This test demonstrates how method calls will attempt to unify an opaque type with a reference
//! if the method takes `&self` as its argument. This is almost never what is desired, as the user
//! would like to have method resolution happen on the opaque type instead of inferring the hidden
//! type. Once type-alias-impl-trait requires annotating which functions should constrain the hidden
//! type, this won't be as much of a problem, as most functions that do method calls on opaque types
//! won't also be the ones defining the hidden type.

//@ revisions: current next
//@[next] compile-flags: -Znext-solver

#![feature(type_alias_impl_trait)]

pub struct Foo {
    bar: Tait,
}

type Tait = impl Iterator<Item = ()>;

impl Foo {
    #[define_opaque(Tait)]
    pub fn new() -> Foo {
        Foo { bar: std::iter::empty() }
    }

    #[define_opaque(Tait)]
    fn foo(&mut self) {
        //[current]~^ ERROR: item does not constrain
        self.bar.next().unwrap();
        //[next]~^ ERROR: type annotations needed
    }
}

fn main() {}
