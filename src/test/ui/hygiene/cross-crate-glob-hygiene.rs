// Check that globs cannot import hygienic identifiers from a macro expansion
// in another crate. `my_struct` is a `macro_rules` macro, so the struct it
// defines is only not imported because `my_struct` is defined by a macros 2.0
// macro.

// aux-build:use_by_macro.rs

extern crate use_by_macro;

use use_by_macro::*;

mod m {
    use use_by_macro::*;

    my_struct!(define);
}

use m::*;

fn main() {
    let x = my_struct!(create);
    //~^ ERROR cannot find struct, variant or union type `MyStruct` in this scope
}
