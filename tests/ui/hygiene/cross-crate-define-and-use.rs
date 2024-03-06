// Check that a marco from another crate can define an item in one expansion
// and use it from another, without it being visible to everyone.
// This requires that the definition of `my_struct` preserves the hygiene
// information for the tokens in its definition.

//@ check-pass
//@ aux-build:use_by_macro.rs

extern crate use_by_macro;

use use_by_macro::*;

enum MyStruct {}
my_struct!(define);

fn main() {
    let x = my_struct!(create);
}
