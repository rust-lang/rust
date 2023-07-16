// Regression test to ensure that both `AtomicU8` items are displayed.

#![crate_name = "foo"]

// @has 'foo/index.html'
// @has - '//*[@class="item-name"]/a[@class="struct"]' 'AtomicU8'
// @has - '//*[@class="item-name"]/a[@class="constant"]' 'AtomicU8'
// @has - '//*[@id="reexport.AtomicU8"]/code' 'pub use crate::thing::AtomicU8;'

mod thing {
    pub use std::sync::atomic::AtomicU8;

    #[allow(non_upper_case_globals)]
    pub const AtomicU8: () = ();
}

pub use crate::thing::AtomicU8;
