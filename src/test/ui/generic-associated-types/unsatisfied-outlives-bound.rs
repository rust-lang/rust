#![allow(incomplete_features)]
#![feature(generic_associated_types)]

trait ATy {
    type Item<'a>: 'a;
}

impl<'b> ATy for &'b () {
    type Item<'a> = &'b ();
    //~^ ERROR does not fulfill the required lifetime
}

trait StaticTy {
    type Item<'a>: 'static;
}

impl StaticTy for () {
    type Item<'a> = &'a ();
    //~^ ERROR does not fulfill the required lifetime
}

fn main() {}
