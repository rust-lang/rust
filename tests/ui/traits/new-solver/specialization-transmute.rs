// compile-flags: -Ztrait-solver=next

#![feature(specialization)]
//~^ WARN the feature `specialization` is incomplete

trait Default {
   type Id;

   fn intu(&self) -> &Self::Id;
}

impl<T> Default for T {
   default type Id = T;

   fn intu(&self) -> &Self::Id {
        self
        //~^ ERROR cannot satisfy `T <: <T as Default>::Id`
   }
}

fn transmute<T: Default<Id = U>, U: Copy>(t: T) -> U {
    *t.intu()
}

use std::num::NonZeroU8;
fn main() {
    let s = transmute::<u8, Option<NonZeroU8>>(0);
    //~^ ERROR cannot satisfy `<u8 as Default>::Id == Option<NonZeroU8>
    assert_eq!(s, None);
}
