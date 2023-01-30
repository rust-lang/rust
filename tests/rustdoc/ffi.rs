// aux-build:rustdoc-ffi.rs
// ignore-cross-compile

extern crate rustdoc_ffi as lib;

// @has ffi/fn.foreigner.html //pre 'pub unsafe extern "C" fn foreigner(cold_as_ice: u32)'
pub use lib::foreigner;

extern "C" {
    // @has ffi/fn.another.html //pre 'pub unsafe extern "C" fn another(cold_as_ice: u32)'
    pub fn another(cold_as_ice: u32);
}
