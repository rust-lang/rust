// Check for unused crate dep, no path

// edition:2018
// compile-flags: -Dunused-crate-dependencies  -Zunstable-options --json unused-externs --error-format=json
// aux-crate:bar=bar.rs

fn main() {}
