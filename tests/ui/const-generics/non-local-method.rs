//@ compile-flags: --crate-type lib
//@ aux-build:non_local.rs

extern crate non_local;

use non_local::NonLocal;

struct Local;

impl NonLocal for Local {
    fn method<const N: u16>() {
        //~^ ERROR: associated function `method` has an incompatible type for const generic parameter [E0053]
        print!("hello\n");
    }
}
