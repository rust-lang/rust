// Regression test for <https://github.com/rust-lang/rust/issues/101129>.

#![feature(doc_auto_cfg)]
#![crate_type = "lib"]
#![crate_name = "foo"]

pub struct S;
pub trait MyTrait1 {}
pub trait MyTrait2 {}

// @has foo/struct.S.html
// @has - '//*[@id="impl-MyTrait1-for-S"]//*[@class="stab portability"]' \
//        'Available on non-crate feature coolstuff only.'
#[cfg(not(feature = "coolstuff"))]
impl MyTrait1 for S {}

#[cfg(not(feature = "coolstuff"))]
mod submod {
    use crate::{S, MyTrait2};
    // This impl should also have the `not(feature = "coolstuff")`.
    // @has - '//*[@id="impl-MyTrait2-for-S"]//*[@class="stab portability"]' \
    //        'Available on non-crate feature coolstuff only.'
    impl MyTrait2 for S {}
}
