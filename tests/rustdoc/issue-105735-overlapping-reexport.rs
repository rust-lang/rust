// Regression test to ensure that both `AtomicU8` items are displayed but not the re-export.

#![crate_name = "foo"]
#![no_std]

// @has 'foo/index.html'
// @has - '//*[@class="item-name"]/a[@class="struct"]' 'AtomicU8'
// @has - '//*[@class="item-name"]/a[@class="constant"]' 'AtomicU8'
// We also ensure we don't have another item displayed.
// @count - '//*[@id="main-content"]/*[@class="small-section-header"]' 2
// @has - '//*[@id="main-content"]/*[@class="small-section-header"]' 'Structs'
// @has - '//*[@id="main-content"]/*[@class="small-section-header"]' 'Constants'

mod thing {
    pub use core::sync::atomic::AtomicU8;

    #[allow(non_upper_case_globals)]
    pub const AtomicU8: () = ();
}

pub use crate::thing::AtomicU8;
