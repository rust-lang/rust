// Check that we correctly prevent users from making trait objects
// from traits with generic methods, unless `where Self : Sized` is
// present.
//@ revisions: curr dyn_compatible_for_dispatch

#![cfg_attr(dyn_compatible_for_dispatch, feature(dyn_compatible_for_dispatch))]


trait Bar {
    fn bar<T>(&self, t: T);
}

trait Quux {
    fn bar<T>(&self, t: T)
        where Self : Sized;
}

fn make_bar<T:Bar>(t: &T) -> &dyn Bar {
    //[curr]~^ ERROR E0038
    t
    //[dyn_compatible_for_dispatch]~^ ERROR E0038
    //[curr]~^^ ERROR E0038
}

fn make_bar_explicit<T:Bar>(t: &T) -> &dyn Bar {
    //[curr]~^ ERROR E0038
    t as &dyn Bar
    //[dyn_compatible_for_dispatch]~^ ERROR E0038
    //[curr]~^^ ERROR E0038
    //[curr]~| ERROR E0038
}

fn make_quux<T:Quux>(t: &T) -> &dyn Quux {
    t
}

fn make_quux_explicit<T:Quux>(t: &T) -> &dyn Quux {
    t as &dyn Quux
}

fn main() {
}
