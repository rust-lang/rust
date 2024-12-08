//! Check that projections will constrain opaque types while looking for
//! matching impls and error if ambiguous.

//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver

#![feature(type_alias_impl_trait)]

struct Foo;

type Bar = impl Sized;

trait Trait<T> {
    type Assoc: Default;
}

impl Trait<()> for Foo {
    type Assoc = u32;
}

impl Trait<u32> for Foo {
    type Assoc = u32;
}

fn bop(_: Bar) {
    let x = <Foo as Trait<Bar>>::Assoc::default();
    //[next]~^ ERROR: cannot satisfy `Foo: Trait<Bar>`
    //[current]~^^ ERROR: `Foo: Trait<Bar>` is not satisfied
}

fn main() {}
