//@ compile-flags: -Znext-solver
#![feature(specialization)]
//~^ WARN the feature `specialization` is incomplete

trait Default {
    type Id;

    fn intu(&self) -> &Self::Id;
}

impl<T> Default for T {
    default type Id = T;
    // This will be fixed by #111994
    fn intu(&self) -> &Self::Id {
        //~^ ERROR type annotations needed
        //~| ERROR cannot normalize `<T as Default>::Id: '_`
        self //~ ERROR cannot satisfy
    }
}

fn transmute<T: Default<Id = U>, U: Copy>(t: T) -> U {
    *t.intu()
}

use std::num::NonZero;

fn main() {
    let s = transmute::<u8, Option<NonZero<u8>>>(0); //~ ERROR cannot satisfy
    assert_eq!(s, None);
}
