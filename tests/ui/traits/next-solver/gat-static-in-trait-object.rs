//! regression test from https://github.com/rust-lang/rust/pull/160831/changes#r3852248101
//! once we allow GATs in object types, we want to make sure this isn't unsound.

//@ compile-flags: -Znext-solver

use std::any::Any;

trait Trait {
    type Assoc<'a>: 'a;
}

fn tr<T: Trait>(x: T::Assoc<'static>) -> Box<dyn Any> { Box::new(x) }

fn foo<'s>(x: &'s str) -> Box<dyn Any>
where
    dyn for<'hr> Trait<Assoc<'hr> = &'s str>: Trait<Assoc<'static> = &'s str>,
    //~^ ERROR the trait `Trait` is not dyn compatible
    //~| ERROR the trait `Trait` is not dyn compatible
{
    tr::<dyn for<'hr> Trait<Assoc<'hr> = &'s str>>(x)
    //~^ ERROR the trait `Trait` is not dyn compatible
}

fn main() {}
