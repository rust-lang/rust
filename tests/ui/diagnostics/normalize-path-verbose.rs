//@ compile-flags: --verbose

// Check that `--verbose` prints diagnostic paths as given, without lexical
// normalization. See #51349.
#[path = "auxiliary/sub/mod.rs"]
mod sub;

fn main() {}

//~? ERROR mismatched types
