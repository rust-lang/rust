// Regression test for <https://github.com/rust-lang/rust/issues/154921>.
// This test ensures that auto-generated and explicit `doc(cfg)` attributes are correctly
// preserved for locally re-exported type aliases.

//@ compile-flags: --cfg feature="foo"

#![crate_name = "foo"]
#![feature(doc_cfg)]

mod inner {
    #[cfg(feature = "foo")]
    pub type One = u32;

    #[doc(cfg(feature = "foo"))]
    pub type Two = u32;
}

//@ has 'foo/index.html'
// There should be two items in the type aliases table.
//@ count - '//*[@class="item-table"]/dt' 2
// Both of them should have the portability badge in the module index.
//@ count - '//*[@class="item-table"]/dt/*[@class="stab portability"]' 2

//@ has 'foo/type.One.html'
// Check that the individual type page has the portability badge.
//@ count - '//*[@id="main-content"]/*[@class="item-info"]/*[@class="stab portability"]' 1
//@ has - '//*[@id="main-content"]/*[@class="item-info"]/*[@class="stab portability"]' 'foo'

//@ has 'foo/type.Two.html'
// Check the explicit doc(cfg) type page as well.
//@ count - '//*[@id="main-content"]/*[@class="item-info"]/*[@class="stab portability"]' 1
//@ has - '//*[@id="main-content"]/*[@class="item-info"]/*[@class="stab portability"]' 'foo'

pub use self::inner::{One, Two};
