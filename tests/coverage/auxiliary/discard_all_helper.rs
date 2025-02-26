//@ edition: 2021

// Force this function to be generated in its home crate, so that it ends up
// with normal coverage metadata.
#[inline(never)]
pub fn external_function() {}
