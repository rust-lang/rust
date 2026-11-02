#![warn(unused_macros)]
// #57966
mod has_a_macro {
//~^ HELP: consider annotating the enclosing module with `#[macro_use]`
    macro_rules! a_macro { () => {} }
    //~^ WARN: unused macro definition
}

use crate::has_a_macro::a_macro;
//~^ ERROR: unresolved import `crate::has_a_macro::a_macro`

fn main() {
    a_macro!();
}
