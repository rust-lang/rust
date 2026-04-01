// Regression test for <https://github.com/rust-lang/rust/issues/60522>.
// This test ensures that the `banana` and `peach` modules don't appear twice
// and that the visible modules are not the re-exported ones.

#![crate_name = "foo"]

//@ has 'foo/index.html'
//@ count - '//*[@id="main-content"]/*[@class="section-header"]' 1
//@ has - '//*[@id="main-content"]/*[@class="section-header"]' 'Modules'
//@ count - '//*[@id="main-content"]/*[@class="item-table"]//*[@class="mod"]' 2
//@ has - '//*[@id="main-content"]//*[@class="mod"]' 'banana'
//@ has - '//*[@id="main-content"]//*[@href="banana/index.html"]' 'banana'
//@ has - '//*[@id="main-content"]//*[@class="mod"]' 'peach'
//@ has - '//*[@id="main-content"]//*[@href="peach/index.html"]' 'peach'

pub use crate::my_crate::*;

mod my_crate {
    pub mod banana {
        pub struct Yellow;
    }
    pub mod peach {
        pub struct Pink;
    }
}

//@ has 'foo/banana/index.html'
//@ count - '//*[@id="main-content"]//*[@class="struct"]' 1
//@ has - '//*[@id="main-content"]//*[@class="struct"]' 'Brown'
pub mod banana {
    pub struct Brown;
}

//@ has 'foo/peach/index.html'
//@ count - '//*[@id="main-content"]//*[@class="struct"]' 1
//@ has - '//*[@id="main-content"]//*[@class="struct"]' 'Pungent'
pub mod peach {
    pub struct Pungent;
}
