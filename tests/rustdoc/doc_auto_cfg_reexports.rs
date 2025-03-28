// Checks that `cfg` are correctly applied on inlined reexports.

#![crate_name = "foo"]

// Check with `std` item.
//@ has 'foo/index.html' '//*[@class="stab portability"]' 'Non-moustache'
//@ has 'foo/struct.C.html' '//*[@class="stab portability"]' \
//      'Available on non-crate feature moustache only.'
#[cfg(not(feature = "moustache"))]
pub use std::cell::RefCell as C;

// Check with local item.
mod x {
    pub struct B;
}

//@ has 'foo/index.html' '//*[@class="stab portability"]' 'Non-pistache'
//@ has 'foo/struct.B.html' '//*[@class="stab portability"]' \
//      'Available on non-crate feature pistache only.'
#[cfg(not(feature = "pistache"))]
pub use crate::x::B;
