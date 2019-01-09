// aux-build:import_crate_var.rs

// compile-pass
// skip-codegen
#[macro_use] extern crate import_crate_var;


fn main() {
    m!();
    //~^ WARN `$crate` may not be imported
    //~| NOTE `use $crate;` was erroneously allowed and will become a hard error
}
