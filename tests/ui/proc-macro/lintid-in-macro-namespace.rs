//@ check-pass
//@ proc-macro: lint-id.rs

extern crate lint_id;

// macro namespace
pub use lint_id::ambiguous_thing;

// type namespace
#[allow(non_camel_case_types)]
pub struct ambiguous_thing {}

// value namespace
pub fn ambiguous_thing() {}

fn main() {}
