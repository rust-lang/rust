// Regression test to ensure that both `AtomicU8` items are displayed but not the re-export.
// https://github.com/rust-lang/rust/issues/105735

#![crate_name = "foo"]
#![no_std]

//@ has 'foo/index.html'
//@ has - '//dt/a[@class="type"]' 'AtomicU8'
//@ has - '//dt/a[@class="constant"]' 'AtomicU8'
// We also ensure we don't have another item displayed.
//@ count - '//*[@id="main-content"]/*[@class="section-header"]' 2
//@ has - '//*[@id="main-content"]/*[@class="section-header"]' 'Type Aliases'
//@ has - '//*[@id="main-content"]/*[@class="section-header"]' 'Constants'

mod other {
    pub type AtomicU8 = ();
}

mod thing {
    pub use crate::other::AtomicU8;

    #[allow(non_upper_case_globals)]
    pub const AtomicU8: () = ();
}

pub use crate::thing::AtomicU8;
