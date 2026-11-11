//! Tests the error reporting when a declared module file is missing.
mod missing_mod_inline;
fn main() {}

//~? ERROR file not found for module `missing`
