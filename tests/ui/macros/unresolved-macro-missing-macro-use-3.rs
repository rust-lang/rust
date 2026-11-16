#![warn(unused_macros)]
// #57966
#[macro_use]
mod has_a_macro {
    macro_rules! a_macro { () => {} }
    //~^ NOTE: `a_macro` could refer to the macro defined here
}

use crate::has_a_macro::a_macro;
//~^ ERROR: unresolved import `crate::has_a_macro::a_macro`
//~| NOTE: `a_macro` could also refer to the unresolved item imported here
//~| NOTE: no `a_macro` in `has_a_macro`

fn main() {
    a_macro!();
    //~^ ERROR: `a_macro` is ambiguous
    //~| NOTE: ambiguous name
    //~| NOTE: ambiguous because of a conflict between a `macro_rules` name and a non-`macro_rules` name from another module
}
