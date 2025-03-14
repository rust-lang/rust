// Show diagnostics for unbalanced parens.
//@ compile-flags: -Zcrate-attr=(
//@ error-pattern:unclosed delimiter
fn main() {}
