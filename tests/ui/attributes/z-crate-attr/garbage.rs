// Show diagnostics for invalid tokens
//@ compile-flags: -Zcrate-attr=`%~@$#

fn main() {}

//~? ERROR unknown start of token: `
//~? ERROR expected identifier, found `%`
