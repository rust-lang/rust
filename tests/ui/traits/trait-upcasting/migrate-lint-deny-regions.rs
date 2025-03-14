//@ check-pass
#![warn(deref_into_dyn_supertrait)]

use std::ops::Deref;

trait Bar<'a> {}
trait Foo<'a>: Bar<'a> {}

impl<'a> Deref for dyn Foo<'a> {
    //~^ warn: this `Deref` implementation is covered by an implicit supertrait coercion
    type Target = dyn Bar<'a>;

    fn deref(&self) -> &Self::Target {
        todo!()
    }
}

fn main() {}
