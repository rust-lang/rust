// edition:2018

// Tests that `meta` is whitelisted, even if the crate doesn't exist
// yet (i.e., it causes a different error than `not-whitelisted.rs`).
use meta; //~ ERROR can't find crate for `meta`

fn main() {}
