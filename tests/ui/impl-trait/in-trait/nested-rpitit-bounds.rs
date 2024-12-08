//@ check-pass

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
