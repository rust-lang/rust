mod glob_ok {
    pub mod something {
        pub mod something_else {}
    }
}

mod single_err {}

use glob_ok::*; // glob_ok::something
use single_err::something; //~ ERROR unresolved import `single_err::something`
use something::something_else;

fn main() {}
