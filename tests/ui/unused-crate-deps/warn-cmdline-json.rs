// Check for unused crate dep, warn, json event, expect pass

// edition:2018
// check-pass
// compile-flags: -Wunused-crate-dependencies -Zunstable-options --json unused-externs --error-format=json
// aux-crate:bar=bar.rs

fn main() {}
