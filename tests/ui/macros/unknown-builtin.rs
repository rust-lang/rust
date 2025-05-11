#![feature(rustc_attrs)]

#[rustc_builtin_macro]
macro_rules! unknown { () => () } //~ ERROR cannot find a built-in macro with name `unknown`

// Defining another `line` builtin macro should not cause an error.
#[rustc_builtin_macro]
macro_rules! line { () => () }

fn main() {
    line!();
    std::prelude::v1::line!();
}
