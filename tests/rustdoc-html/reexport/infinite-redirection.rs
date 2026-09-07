#![crate_name = "foo"]

// This test ensures that there is no "infinite redirection" file generated (a
// file which redirects to itself).

// We check it's not a redirection file.
//@ has 'foo/builders/struct.ActionRowBuilder.html'
//@ has - '//*[@id="synthetic-implementations"]' 'Auto Trait Implementations'

// And that the link in the module is targeting it.
//@ has 'foo/builders/index.html'
//@ has - '//a[@href="struct.ActionRowBuilder.html"]' 'ActionRowBuilder'

mod auto {
    mod action_row {
        pub struct ActionRowBuilder;
    }

    #[doc(hidden)]
    pub mod builders {
        pub use super::action_row::ActionRowBuilder;
    }
}

pub use auto::*;

pub mod builders {
    pub use crate::auto::builders::*;
}
