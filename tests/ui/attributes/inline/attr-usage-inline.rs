//! Check that `#[inline]` attribute can only be applied to fn-like targets (e.g. function or
//! closure), and when misapplied to other targets an error is emitted.

#[inline]
fn f() {}

#[inline] //~ ERROR: attribute cannot be used on
struct S;

struct I {
    #[inline]
    i: u8,
}

#[macro_export]
#[inline]
macro_rules! m_e {
    () => {};
}

#[inline] //~ ERROR: attribute should be applied to function or closure
macro_rules! m {
    () => {};
}

fn main() {}
