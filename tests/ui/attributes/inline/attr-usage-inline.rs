//! Check that `#[inline]` attribute can only be applied to fn-like targets (e.g. function or
//! closure), and when misapplied to other targets an error is emitted.

#[inline]
fn f() {}

#[inline] //~ ERROR: attribute cannot be used on
struct S;

struct I {
    #[inline]
    //~^ WARN attribute cannot be used on
    //~| WARN previously accepted
    i: u8,
}

#[macro_export]
#[inline]
//~^ WARN attribute cannot be used on
//~| WARN previously accepted
macro_rules! m_e {
    () => {};
}

#[inline] //~ ERROR: attribute should be applied to function or closure
//~^ WARN attribute cannot be used on
//~| WARN previously accepted
macro_rules! m {
    () => {};
}

fn main() {}
