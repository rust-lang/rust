// https://github.com/rust-lang/rust/issues/124490

mod a {
    pub mod b {
        pub mod c {}
    }
}

use a::*;

use b::c;
//~^ ERROR: cannot determine resolution for the import
//~| ERROR: cannot determine resolution for the import
//~| ERROR: unresolved import `b::c`
use c as b;

fn main() {}
