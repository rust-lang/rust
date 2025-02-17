//! Check that projections will constrain opaque types while looking for
//! matching impls.

//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver
//@[next]check-pass

#![feature(type_alias_impl_trait)]

struct Foo;

type Bar = impl Sized;

trait Trait<T> {
    type Assoc: Default;
}

impl Trait<()> for Foo {
    type Assoc = u32;
}

fn bop(_: Bar) {
    let x = <Foo as Trait<Bar>>::Assoc::default();
    //[current]~^ `Foo: Trait<Bar>` is not satisfied
    //[current]~| `Foo: Trait<Bar>` is not satisfied
}

fn main() {}
