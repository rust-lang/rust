//@ check-pass

use std::ops::Deref;

trait Foo {
    fn foo() -> impl Deref<Target = impl Deref<Target = impl Sized>> {
        &&()
    }
}

fn main() {}
