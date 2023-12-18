#![warn(clippy::wildcard_imports)]

mod utils {
    pub fn print() {}
}

mod utils_plus {
    pub fn do_something() {}
}

use utils::*;
use utils_plus::*;
//~^ ERROR: usage of wildcard import

fn main() {
    print();
    do_something();
}
