//@ check-pass
//@ proc-macro: makro.rs
//@ edition: 2021
//@ ignore-backends: gcc

// Check that a proc-macro doesn't try to parse frontmatter and instead treats
// it as a regular Rust token sequence. See `auxiliary/makro.rs` for details.

makro::check!();

fn main() {}
