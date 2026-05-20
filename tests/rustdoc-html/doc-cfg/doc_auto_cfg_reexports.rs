// Checks that `cfg` are correctly applied on inlined reexports.

#![crate_name = "foo"]
#![feature(doc_cfg)]

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

// Now checking that `cfg`s are not applied on non-inlined reexports.
pub mod pub_sub_mod {
    //@ has 'foo/pub_sub_mod/index.html'
    // There should be only only item with `cfg` note.
    //@ count - '//*[@class="stab portability"]' 1
    // And obviously the item should be "blabla".
    //@ has - '//dt' 'blablaNon-pistache'
    #[cfg(not(feature = "pistache"))]
    pub fn blabla() {}

    pub use self::blabla as another;
}
