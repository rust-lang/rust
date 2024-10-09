// Check that we correctly prevent users from making trait objects
// from traits where `Self : Sized`.
//
//@ revisions: curr dyn_compatible_for_dispatch

#![cfg_attr(dyn_compatible_for_dispatch, feature(dyn_compatible_for_dispatch))]

trait Bar: Sized {
    fn bar<T>(&self, t: T);
}

fn make_bar<T: Bar>(t: &T) -> &dyn Bar {
    //[curr]~^ ERROR E0038
    t
    //~^ ERROR E0038
}

fn main() {}
