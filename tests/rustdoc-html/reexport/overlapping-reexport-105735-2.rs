// Regression test to ensure that both `Thing` items are displayed but not the re-export.
// https://github.com/rust-lang/rust/issues/105735

#![crate_name = "foo"]
#![no_std]

//@ has 'foo/index.html'
//@ has - '//dt/a[@class="type"]' 'Thing'
//@ has - '//dt/a[@class="constant"]' 'Thing'
// We also ensure we don't have another item displayed.
//@ count - '//*[@id="main-content"]/*[@class="section-header"]' 2
//@ has - '//*[@id="main-content"]/*[@class="section-header"]' 'Type Aliases'
//@ has - '//*[@id="main-content"]/*[@class="section-header"]' 'Constants'

mod other {
    pub type Thing = ();
}

mod thing {
    pub use crate::other::Thing;

    #[allow(non_upper_case_globals)]
    pub const Thing: () = ();
}

pub use crate::thing::Thing;
