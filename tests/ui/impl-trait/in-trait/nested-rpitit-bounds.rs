//@ check-pass
//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver

use std::ops::Deref;

trait Foo {
    fn foo() -> impl Deref<Target = impl Deref<Target = impl Sized>> {
        &&()
    }

    fn bar() -> impl Deref<Target = Option<impl Sized>> {
        &Some(())
    }
}

fn main() {}
