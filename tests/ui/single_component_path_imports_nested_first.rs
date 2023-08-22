#![warn(clippy::single_component_path_imports)]
#![allow(unused_imports)]
//@no-rustfix
use regex;
//~^ ERROR: this import is redundant
//~| NOTE: `-D clippy::single-component-path-imports` implied by `-D warnings`

use serde as edres;

pub use serde;

fn main() {
    regex::Regex::new(r"^\d{4}-\d{2}-\d{2}$").unwrap();
}

mod root_nested_use_mod {
    use {regex, serde};
    //~^ ERROR: this import is redundant
    //~| ERROR: this import is redundant
    #[allow(dead_code)]
    fn root_nested_use_mod() {}
}
