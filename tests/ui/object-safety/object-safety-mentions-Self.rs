// Check that we correctly prevent users from making trait objects
// form traits that make use of `Self` in an argument or return
// position, unless `where Self : Sized` is present..
//
//@ revisions: curr object_safe_for_dispatch

#![cfg_attr(object_safe_for_dispatch, feature(object_safe_for_dispatch))]


trait Bar {
    fn bar(&self, x: &Self);
}

trait Baz {
    fn baz(&self) -> Self;
}

trait Quux {
    fn quux(&self, s: &Self) -> Self where Self : Sized;
}

fn make_bar<T:Bar>(t: &T) -> &dyn Bar {
    //[curr]~^ ERROR E0038
    t
    //~^ ERROR E0038
}

fn make_baz<T:Baz>(t: &T) -> &dyn Baz {
    //[curr]~^ ERROR E0038
    t
    //~^ ERROR E0038
}

fn make_quux<T:Quux>(t: &T) -> &dyn Quux {
    t
}

fn make_quux_explicit<T:Quux>(t: &T) -> &dyn Quux {
    t as &dyn Quux
}

fn main() {}
