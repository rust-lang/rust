// Regression test for <https://github.com/rust-lang/rust/issues/98007>.

#![feature(no_core)]
#![no_core]

mod auto {
    mod action_row {
        pub struct ActionRowBuilder;
    }

    #[doc(hidden)]
    pub mod builders {
        pub use super::action_row::ActionRowBuilder;
    }
}

// @count "$.index[*][?(@.name=='builders')]" 2
pub use auto::*;

pub mod builders {
    pub use crate::auto::builders::*;
}
