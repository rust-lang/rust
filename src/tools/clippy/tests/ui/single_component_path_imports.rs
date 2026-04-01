#![warn(clippy::single_component_path_imports)]
#![allow(unused_imports)]

use core;

use regex;
//~^ single_component_path_imports

use serde as edres;

pub use serde;

use std;

macro_rules! m {
    () => {
        use regex;
    };
}

fn main() {
    regex::Regex::new(r"^\d{4}-\d{2}-\d{2}$").unwrap();

    // False positive #5154, shouldn't trigger lint.
    m!();

    // False positive #10549
    let _ = self::std::io::stdout();
    let _ = 0 as self::core::ffi::c_uint;
}

mod hello_mod {
    use regex;
    //~^ single_component_path_imports
    #[allow(dead_code)]
    fn hello_mod() {}
}

mod hi_mod {
    use self::regex::{Regex, RegexSet};
    use regex;
    #[allow(dead_code)]
    fn hi_mod() {}
}
