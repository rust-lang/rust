// Regression test to ensure that both `Ordering` items are displayed but not the re-export.
// https://github.com/rust-lang/rust/issues/105735

#![crate_name = "foo"]
#![no_std]

//@ has 'foo/index.html'
//@ has - '//dt/a[@class="enum"]' 'Ordering'
//@ has - '//dt/a[@class="constant"]' 'Ordering'
// We also ensure we don't have another item displayed.
//@ count - '//*[@id="main-content"]/*[@class="section-header"]' 2
//@ has - '//*[@id="main-content"]/*[@class="section-header"]' 'Enums'
//@ has - '//*[@id="main-content"]/*[@class="section-header"]' 'Constants'

mod thing {
    pub use core::cmp::Ordering;

    #[allow(non_upper_case_globals)]
    pub const Ordering: () = ();
}

pub use crate::thing::Ordering;
