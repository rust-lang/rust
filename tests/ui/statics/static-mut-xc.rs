//@ run-pass
#![allow(non_upper_case_globals)]

// Constants (static variables) can be used to match in patterns, but mutable
// statics cannot. This ensures that there's some form of error if this is
// attempted.

//@ aux-build:static_mut_xc.rs

extern crate static_mut_xc;

unsafe fn static_bound(_: &'static isize) {}

fn static_bound_set(a: &'static mut isize) {
    *a = 3;
}

unsafe fn run() {
    assert_eq!(static_mut_xc::a, 3);
    //~^ WARN creating a shared reference to mutable static is discouraged [static_mut_refs]
    static_mut_xc::a = 4;
    assert_eq!(static_mut_xc::a, 4);
    //~^ WARN creating a shared reference to mutable static is discouraged [static_mut_refs]
    static_mut_xc::a += 1;
    assert_eq!(static_mut_xc::a, 5);
    //~^ WARN creating a shared reference to mutable static is discouraged [static_mut_refs]
    static_mut_xc::a *= 3;
    assert_eq!(static_mut_xc::a, 15);
    //~^ WARN creating a shared reference to mutable static is discouraged [static_mut_refs]
    static_mut_xc::a = -3;
    assert_eq!(static_mut_xc::a, -3);
    //~^ WARN creating a shared reference to mutable static is discouraged [static_mut_refs]
    static_bound(&static_mut_xc::a);
    //~^ WARN shared reference to mutable static is discouraged [static_mut_refs]
    static_bound_set(&mut static_mut_xc::a);
    //~^ WARN mutable reference to mutable static is discouraged [static_mut_refs]
}

pub fn main() {
    unsafe { run() }
}

pub mod inner {
    pub static mut a: isize = 4;
}
