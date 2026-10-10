// Show diagnostics for unbalanced parens.
//@ compile-flags: -Zcrate-attr=(
//~? ERROR mismatched closing delimiter
fn main() {}
