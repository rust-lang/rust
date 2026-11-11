//@ edition: 2015
//@ proc-macro: issue-61963.rs
//@ proc-macro: issue-61963-1.rs
#![deny(bare_trait_objects)]

#[macro_use]
extern crate issue_61963;
#[macro_use]
extern crate issue_61963_1;

// This test checks that the bare trait object lint does not trigger on macro attributes that
// generate code which would trigger the lint.

pub struct Baz;
pub trait Bar {}
pub struct Qux<T>(T);

#[dom_struct]
pub struct Foo {
    //~^ ERROR trait objects without an explicit `dyn` are deprecated [bare_trait_objects]
    //~| WARN this is accepted in the current edition
    qux: Qux<Qux<Baz>>,
    bar: Box<Bar>,
    //~^ ERROR trait objects without an explicit `dyn` are deprecated [bare_trait_objects]
    //~| WARN this is accepted in the current edition
}

fn main() {}
