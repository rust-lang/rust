// Check that we correctly prevent users from making trait objects
// from traits where `Self : Sized`.
//
// revisions: curr object_safe_for_dispatch

#![cfg_attr(object_safe_for_dispatch, feature(object_safe_for_dispatch))]

trait Bar
    where Self : Sized
{
    fn bar<T>(&self, t: T);
}

fn make_bar<T:Bar>(t: &T) -> &dyn Bar {
    //[curr]~^ ERROR E0038
    t
    //[object_safe_for_dispatch]~^ ERROR E0038
}

fn main() {
}
