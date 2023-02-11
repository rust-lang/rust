#![deny(deref_into_dyn_supertrait)]

extern crate core;

use core::ops::Deref;

// issue 89190
trait A {}
trait B: A {}

impl<'a> Deref for dyn 'a + B {
    //~^ ERROR `(dyn B + 'a)` implements `Deref` with supertrait `A` as target
    //~| WARN this was previously accepted by the compiler but is being phased out;

    type Target = dyn A;
    fn deref(&self) -> &Self::Target {
        todo!()
    }
}

fn take_a(_: &dyn A) {}

fn whoops(b: &dyn B) {
    take_a(b)
}

fn main() {}
