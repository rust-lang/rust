//@ compile-flags: --crate-type=lib
#![allow(internal_features)]

#[rustc_force_inline]
//~^ ERROR #[rustc_force_inline] forces a free function to be inlined
pub fn bare() {
}

#[rustc_force_inline = "the test requires it"]
//~^ ERROR #[rustc_force_inline] forces a free function to be inlined
pub fn justified() {
}
