// check-pass
// compile-flags:--test

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
