//! This test ICEs because the `repr(packed)` attributes
//! end up on the `Dealigned` struct's attribute list, but the
//! derive didn't see that.
//!
//! Because we now `Fatal.raise()` in resolve when encountering
//! duplicated names, the ICE in #120873 no longer happens.

#[repr(packed)]
struct Dealigned<T>(u8, T);

#[derive(PartialEq)]
#[repr(C)]
struct Dealigned<T>(u8, T); //~ ERROR: the name `Dealigned` is defined multiple times

fn main() {}
