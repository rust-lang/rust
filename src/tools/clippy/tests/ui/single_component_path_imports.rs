// run-rustfix
// edition:2018
#![warn(clippy::single_component_path_imports)]
#![allow(unused_imports)]

use regex;
use serde as edres;
pub use serde;

macro_rules! m {
    () => {
        use regex;
    };
}

fn main() {
    regex::Regex::new(r"^\d{4}-\d{2}-\d{2}$").unwrap();

    // False positive #5154, shouldn't trigger lint.
    m!();
}
