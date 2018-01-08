
#![warn(empty_line_after_outer_attr)]

// This should produce a warning
#[crate_type = "lib"]

fn with_one_newline() { assert!(true) }

// This should produce a warning, too
#[crate_type = "lib"]


fn with_two_newlines() { assert!(true) }

// This should not produce a warning
#[allow(non_camel_case_types)]
#[allow(missing_docs)]
#[allow(missing_docs)]
fn three_attributes() { assert!(true) }

fn main() { }
