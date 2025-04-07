// Show diagnostics for unbalanced parens.
//@ compile-flags: -Zcrate-attr=(

fn main() {}

//~? ERROR this file contains an unclosed delimiter
