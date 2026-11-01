// https://github.com/rust-lang/rust/issues/126389

mod a {
    pub mod b {
        pub mod c {}
    }
}

use a::*;

use b::c;
//~^ ERROR: unresolved import `b::c`
//~| ERROR: cannot determine resolution for the import
//~| ERROR: cannot determine resolution for the import
use c as b;

fn c() {}

fn main() { }
