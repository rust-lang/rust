//@ compile-flags: --crate-type=lib

#[rustc_force_inline]
//~^ ERROR use of an internal attribute [E0658]
//~| NOTE the `rustc_force_inline` attribute is an internal implementation detail that will never be stable
//~| NOTE the `rustc_force_inline` attribute forces a free function to be inlined
pub fn bare() {
}
