// This test ensures that non-glob reexports don't get their attributes merge with
// the reexported item whereas glob reexports do with the `doc_auto_cfg` feature.

#![crate_name = "foo"]
#![feature(doc_cfg)]

//@ has 'foo/index.html'
// There are two items.
//@ count - '//*[@class="item-table"]/dt' 2
// Only one of them should have an attribute.
//@ count - '//*[@class="item-table"]/dt/*[@class="stab portability"]' 1

mod a {
    #[cfg(not(feature = "a"))]
    pub struct Test1;
}

mod b {
    #[cfg(not(feature = "a"))]
    pub struct Test2;
}

//@ has 'foo/struct.Test1.html'
//@ count - '//*[@id="main-content"]/*[@class="item-info"]' 1
//@ has - '//*[@id="main-content"]/*[@class="item-info"]' 'Available on non-crate feature a only.'
pub use a::*;
//@ has 'foo/struct.Test2.html'
//@ count - '//*[@id="main-content"]/*[@class="item-info"]' 0
pub use b::Test2;
