// Ensure that using `;` instead of `:` in a macro fragment specifier
// produces a helpful suggestion.

macro_rules! m {
    ($x;ty) => {}; //~ ERROR missing fragment specifier
}

fn main() {}
