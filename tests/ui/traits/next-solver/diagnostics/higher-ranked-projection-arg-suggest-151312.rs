//! Reporting a mismatched higher-ranked projection bound on a method argument
//! used to ICE with "cannot relate bound region" in the new solver.
//!
//! Regression test for <https://github.com/rust-lang/rust/issues/151312>.
//@ compile-flags: -Znext-solver

pub struct Foo;

fn foo(foos: Option<Foo>) {
    let foos = foos.as_ref();
    Bar.bar(foos);
    //~^ ERROR type mismatch resolving
}

struct Bar;

impl Bar {
    pub fn bar<I>(&self, _iter: I)
    where
        for<'a> &'a I: Trait<Assoc = &'a Foo>,
    {
    }
}

trait Trait {
    type Assoc;
}

impl Trait for Option<Foo> {
    type Assoc = Foo;
}
impl<'a> Trait for Option<&'a Foo> {
    type Assoc = &'a Foo;
}
impl<'a, 'b> Trait for &'a Option<&'b Foo> {
    type Assoc = &'b Foo;
}

fn main() {}
