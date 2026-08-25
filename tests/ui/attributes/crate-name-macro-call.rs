// issue: rust-lang/rust#122001
// Ensure we reject macro calls inside `#![crate_name]` as their result wouldn't get honored anyway.

#![crate_name = concat!("my", "crate")] //~ ERROR attribute value must be a literal

fn main() {}
