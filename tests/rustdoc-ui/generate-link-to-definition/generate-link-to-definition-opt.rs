// This test purpose is to check that the "--generate-link-to-definition"
// option can only be used with HTML generation.

//@ compile-flags: -Zunstable-options --generate-link-to-definition --output-format json
//@ check-pass

pub fn f() {}

//~? WARN `--generate-link-to-definition` option can only be used with HTML output format
