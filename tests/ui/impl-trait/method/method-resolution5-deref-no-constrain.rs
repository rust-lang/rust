//! The recursive method call yields the opaque type. We want
//! to use the impl candidate for `Foo` here without constraining
//! the opaque to `&Foo`.

//@ revisions: current next
//@[next] compile-flags: -Znext-solver
//@[next] check-pass

use std::ops::Deref;
struct Foo;
impl Foo {
    fn method(&self) {}
}
fn via_deref() -> impl Deref<Target = Foo> {
    // Currently errors on stable, but should not
    if false {
        via_deref().method();
    }

    Box::new(Foo)
    //[current]~^ ERROR mismatched types
}
fn main() {}
