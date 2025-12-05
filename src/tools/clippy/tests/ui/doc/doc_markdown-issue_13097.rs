#![deny(clippy::doc_markdown)]
#![allow(clippy::doc_lazy_continuation)]

mod issue13097 {
    // This test checks that words starting with capital letters and ending with "ified" don't
    // trigger the lint.
    pub enum OutputFormat {
        /// HumaNified
        //~^ ERROR: item in documentation is missing backticks
        Plain,
        // Should not warn!
        /// JSONified console output
        Json,
    }
}

#[rustfmt::skip]
pub enum OutputFormat {
    /**
     * HumaNified
     //~^ ERROR: item in documentation is missing backticks
     * Before \u{08888} HumaNified \{u08888} After
     //~^ ERROR: item in documentation is missing backticks
     * meow meow \[meow_meow\] meow meow?
     //~^ ERROR: item in documentation is missing backticks
     * \u{08888} meow_meow \[meow meow] meow?
     //~^ ERROR: item in documentation is missing backticks
     * Above
     * \u{08888}
     * \[hi\](<https://example.com>) HumaNified \[example](<https://example.com>)
     //~^ ERROR: item in documentation is missing backticks
     * \u{08888}
     * Below
     */
    Plain,
    // Should not warn!
    /// JSONified console output
    Json,
}
