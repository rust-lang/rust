// Regression test to ensure that both `AtomicU8` items are displayed but not the re-export.

#![crate_name = "foo"]
#![no_std]

// @has 'foo/index.html'
// @has - '//*[@class="item-name"]/a[@class="type"]' 'AtomicU8'
// @has - '//*[@class="item-name"]/a[@class="constant"]' 'AtomicU8'
// We also ensure we don't have another item displayed.
// @count - '//*[@id="main-content"]/*[@class="small-section-header"]' 2
// @has - '//*[@id="main-content"]/*[@class="small-section-header"]' 'Type Definitions'
// @has - '//*[@id="main-content"]/*[@class="small-section-header"]' 'Constants'

mod other {
    pub type AtomicU8 = ();
}

mod thing {
    pub use crate::other::AtomicU8;

    #[allow(non_upper_case_globals)]
    pub const AtomicU8: () = ();
}

pub use crate::thing::AtomicU8;
