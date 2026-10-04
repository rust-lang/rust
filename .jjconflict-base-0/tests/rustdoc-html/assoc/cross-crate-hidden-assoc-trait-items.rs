// Regression test for issue #95717
// Hide cross-crate `#[doc(hidden)]` associated items in trait impls.

#![crate_name = "dependent"]
//@ edition:2021
//@ aux-crate:dependency=cross-crate-hidden-assoc-trait-items.rs

// The trait `Tr` contains 2 hidden and 2 visisible associated items.
// Instead of checking for the absence of the hidden items, check for the presence of the
// visible items instead and assert that there are *exactly two* associated items
// (by counting the number of `section`s). This is more robust and future-proof.

//@ has dependent/struct.Ty.html
//@ has - '//*[@id="associatedtype.VisibleAssoc"]' 'type VisibleAssoc = ()'
//@ has - '//*[@id="associatedconstant.VISIBLE_ASSOC"]' 'const VISIBLE_ASSOC: ()'
//@ count - '//*[@class="impl-items"]/section' 2

//@ has dependent/trait.Tr.html
//@ has - '//*[@id="associatedtype.VisibleAssoc-1"]' 'type VisibleAssoc = ()'
//@ has - '//*[@id="associatedconstant.VISIBLE_ASSOC-1"]' 'const VISIBLE_ASSOC: ()'
//@ count - '//*[@class="impl-items"]/section' 2

pub use dependency::{Tr, Ty};
