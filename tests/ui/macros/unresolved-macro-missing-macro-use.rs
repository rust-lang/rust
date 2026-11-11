#![warn(unused_macros)]
// #57966
mod has_a_macro {
//~^ HELP: consider annotating the enclosing module with `#[macro_use]`
    macro_rules! a_macro { () => {} }
    //~^ HELP: you might be missing a `#[macro_use]` attribute on a `mod` module or `use` import
    //~| WARN: unused macro definition
}

fn main() {
    a_macro!(); //~ ERROR: cannot find macro `a_macro` in this scope
}
