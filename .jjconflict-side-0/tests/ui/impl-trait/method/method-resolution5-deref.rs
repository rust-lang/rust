//! The recursive method call yields the opaque type. We want
//! to use the trait candidate for `impl Foo` here while not
//! applying it for the `impl Deref`.

//@ revisions: current next
//@[next] compile-flags: -Znext-solver
//@ check-pass

use std::ops::Deref;
trait Foo {
    fn method(&self) {}
}
impl Foo for u32 {}
fn via_deref() -> impl Deref<Target = impl Foo> {
    if false {
        via_deref().method();
    }

    Box::new(1u32)
}

fn via_deref_nested() -> Box<impl Deref<Target = impl Foo>> {
    if false {
        via_deref_nested().method();
    }

    Box::new(Box::new(1u32))
}

fn main() {}
