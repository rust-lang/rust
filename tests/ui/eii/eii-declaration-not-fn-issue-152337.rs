// Regression test for ICE "unexpected sort of node in fn_sig()" (issue #152337).
// When the same name is used for a const and an #[eii] function, the declaration
// was incorrectly resolved to the const, causing fn_sig() to be called on a non-function.
#![feature(extern_item_impls)]

const A: () = ();
#[eii]
fn A() {} //~ ERROR the name `A` is defined multiple times
//~^ ERROR expected function or static, found constant
//~| ERROR expected function or static, found constant

fn main() {}
