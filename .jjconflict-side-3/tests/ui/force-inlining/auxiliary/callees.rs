//@ compile-flags: --crate-type=lib
#![feature(rustc_attrs)]

#[rustc_force_inline = "the test requires it"]
pub fn forced_with_reason() {
}

#[rustc_force_inline]
pub fn forced() {
}
