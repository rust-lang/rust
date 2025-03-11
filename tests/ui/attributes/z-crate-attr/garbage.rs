// Show diagnostics for invalid tokens
//@ compile-flags: -Zcrate-attr=`%~@$#
//@ error-pattern:unknown start of token
fn main() {}
