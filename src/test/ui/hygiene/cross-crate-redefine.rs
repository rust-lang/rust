// Check that items with identical `SyntaxContext` conflict even when that
// context involves a mark from another crate.

// aux-build:use_by_macro.rs

extern crate use_by_macro;

use use_by_macro::*;

my_struct!(define);
//~^ ERROR the name `MyStruct` is defined multiple times
my_struct!(define);

fn main() {}
