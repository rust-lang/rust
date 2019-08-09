// aux-build:lifetime_bound_will_change_warning_lib.rs

// Test that various corner cases cause an error. These are tests
// that used to pass before we tweaked object defaults.

#![allow(dead_code)]
#![allow(unused_variables)]


extern crate lifetime_bound_will_change_warning_lib as lib;

fn just_ref(x: &dyn Fn()) {
}

fn ref_obj(x: &Box<dyn Fn()>) {
    // this will change to &Box<Fn()+'static>...

    // Note: no warning is issued here, because the type of `x` will change to 'static
    if false { ref_obj(x); }
}

fn test1<'a>(x: &'a Box<dyn Fn() + 'a>) {
    // just_ref will stay the same.
    just_ref(&**x)
}

fn test1cc<'a>(x: &'a Box<dyn Fn() + 'a>) {
    // same as test1, but cross-crate
    lib::just_ref(&**x)
}

fn test2<'a>(x: &'a Box<dyn Fn() + 'a>) {
    // but ref_obj will not, so warn.
    ref_obj(x) //~ ERROR mismatched types
}

fn test2cc<'a>(x: &'a Box<dyn Fn() + 'a>) {
    // same as test2, but cross crate
    lib::ref_obj(x) //~ ERROR mismatched types
}

fn test3<'a>(x: &'a Box<dyn Fn() + 'static>) {
    // here, we have a 'static bound, so even when ref_obj changes, no error results
    ref_obj(x)
}

fn test3cc<'a>(x: &'a Box<dyn Fn() + 'static>) {
    // same as test3, but cross crate
    lib::ref_obj(x)
}


fn main() {
}
