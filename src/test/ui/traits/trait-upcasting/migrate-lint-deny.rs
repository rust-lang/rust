#![deny(deref_into_dyn_supertrait)]

extern crate core;

use core::ops::Deref;

// issue 89190
trait A {}
trait B: A {}
impl<'a> Deref for dyn 'a + B {
    type Target = dyn A;
    fn deref(&self) -> &Self::Target {
        todo!()
    }
}

fn take_a(_: &dyn A) {}

fn whoops(b: &dyn B) {
    take_a(b)
    //~^ ERROR `dyn B` implements `Deref` with supertrait `(dyn A + 'static)` as output
    //~^^ WARN this was previously accepted by the compiler but is being phased out;
}

fn main() {}
