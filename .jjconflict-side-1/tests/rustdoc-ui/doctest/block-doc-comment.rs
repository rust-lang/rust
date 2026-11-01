//@ check-pass
//@ compile-flags:--test
//@ normalize-stdout: "finished in \d+\.\d+s" -> "finished in $$TIME"

// This test ensures that no code block is detected in the doc comments.

pub mod Wormhole {
    /** # Returns
     *
     */
    pub fn foofoo() {}
    /**
     * # Returns
     *
     */
    pub fn barbar() {}
}
