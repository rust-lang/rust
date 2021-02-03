// edition:2018
// aux-build:external_macro.rs

// Ensure that CONST_ERR lint errors
// are not silenced in external macros.
// https://github.com/rust-lang/rust/issues/65300

extern crate external_macro;
use external_macro::static_assert;

fn main() {
    static_assert!(2 + 2 == 5); //~ ERROR
    //~| WARN this was previously accepted by the compiler but is being phased out
}
