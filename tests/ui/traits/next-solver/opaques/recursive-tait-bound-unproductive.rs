//@ compile-flags: -Znext-solver
//@ check-fail

#![feature(type_alias_impl_trait)]

trait Trait {}

type Foo = impl Trait;

struct Bar<T>(T);

// This cycle is entirely inside the proof of the opaque's item bound.
// Crossing `OpaqueTypeBound` must not make this recursive impl productive.
impl<T> Trait for Bar<T>
where
    Bar<T>: Trait,
{}

#[define_opaque(Foo)]
fn foo() -> Foo {
    //~^ ERROR overflow evaluating the requirement `Foo == Bar<()>`
    Bar(())
}

fn main() {}
