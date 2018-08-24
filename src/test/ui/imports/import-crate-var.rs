// aux-build:import_crate_var.rs

#![feature(rustc_attrs)]

#[macro_use] extern crate import_crate_var;

#[rustc_error]
fn main() { //~ ERROR compilation successful
    m!();
    //~^ WARN `$crate` may not be imported
    //~| NOTE `use $crate;` was erroneously allowed and will become a hard error
}
