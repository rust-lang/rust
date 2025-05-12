//@ compile-flags: --crate-type=lib
//@ check-pass
#![feature(const_precise_live_drops)]

// Mutable reference allows only mutation of !Drop place.
pub const fn f() {
    let mut x: (Option<String>, u32) = (None, 0);
    let mut a = 10;
    *(&mut a) = 11;
    x.1 = a;
}

// Mutable reference allows only mutation of !Drop place.
pub const fn g() {
    let mut a: (u32, Option<String>) = (0, None);
    let _ = &mut a.0;
}

// Shared reference does not allow for mutation.
pub const fn h() {
    let x: Option<String> = None;
    let _ = &x;
}
