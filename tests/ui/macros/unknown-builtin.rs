// error-pattern: attempted to define built-in macro more than once

#![feature(rustc_attrs)]

#[rustc_builtin_macro]
macro_rules! unknown { () => () } //~ ERROR cannot find a built-in macro with name `unknown`

#[rustc_builtin_macro]
macro_rules! line { () => () } //~ NOTE previously defined here

fn main() {
    line!();
    std::prelude::v1::line!();
}
