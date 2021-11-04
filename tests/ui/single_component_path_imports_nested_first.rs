#![warn(clippy::single_component_path_imports)]
#![allow(unused_imports)]

use regex;
use serde as edres;
pub use serde;

fn main() {
    regex::Regex::new(r"^\d{4}-\d{2}-\d{2}$").unwrap();
}

mod root_nested_use_mod {
    use {regex, serde};
    #[allow(dead_code)]
    fn root_nested_use_mod() {}
}
