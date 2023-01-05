#![allow(dead_code)]

#[inline]
fn f() {}

#[inline] //~ ERROR: attribute should be applied to function or closure
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
