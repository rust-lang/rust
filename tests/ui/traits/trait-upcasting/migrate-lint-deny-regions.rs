#![deny(deref_into_dyn_supertrait)]

use std::ops::Deref;

trait Bar<'a> {}
trait Foo<'a>: Bar<'a> {}

impl<'a> Deref for dyn Foo<'a> {
    //~^ ERROR dyn Foo<'_>` implements `Deref` with supertrait `Bar<'_>` as target
    //~| WARN this was previously accepted by the compiler
    type Target = dyn Bar<'a>;

    fn deref(&self) -> &Self::Target {
        todo!()
    }
}

fn main() {}
