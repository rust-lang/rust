// https://github.com/rust-lang/rust/issues/126376

mod a {
    pub mod b {
        pub trait C {}
    }
}

use a::*;

use e as b;

use b::C as e;
//~^ ERROR: unresolved import `b::C`
//~| ERROR: cannot determine resolution for the import
//~| ERROR: cannot determine resolution for the import

fn e() {}

fn main() { }
