// compile-flags: --crate-type=lib
// check-pass
#![feature(const_mut_refs)]
#![feature(const_precise_live_drops)]

pub const fn f() {
    let mut x: (Option<String>, u32) = (None, 0);
    let mut a = 10;
    *(&mut a) = 11;
    x.1 = a;
}

pub const fn g() {
    let mut a: (u32, Option<String>) = (0, None);
    let _ = &mut a.0;
}
