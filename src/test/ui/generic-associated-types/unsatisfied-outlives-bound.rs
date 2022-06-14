#![feature(generic_associated_types)]

trait ATy {
    type Item<'a>: 'a;
}

impl<'b> ATy for &'b () {
    type Item<'a> = &'b ();
    //~^ ERROR lifetime bound not satisfied
}

trait StaticTy {
    type Item<'a>: 'static;
}

impl StaticTy for () {
    type Item<'a> = &'a ();
    //~^ ERROR lifetime bound not satisfied
}

fn main() {}
