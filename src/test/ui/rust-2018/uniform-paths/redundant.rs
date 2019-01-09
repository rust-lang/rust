// run-pass
// edition:2018

use std;
use std::io;

mod foo {
    pub use std as my_std;
}

mod bar {
    pub use std::{self};
}

fn main() {
    io::stdout();
    self::std::io::stdout();
    foo::my_std::io::stdout();
    bar::std::io::stdout();
}
