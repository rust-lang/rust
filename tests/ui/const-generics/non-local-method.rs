//@ compile-flags: --crate-type lib
//@ aux-build:non_local.rs
//@ edition: 2024

extern crate non_local;

use non_local::NonLocal;

struct Local;

impl NonLocal for Local {
    fn method<const N: u16>() {
        //~^ ERROR: associated function `method` has an incompatible type for const generic parameter [E0053]
        print!("hello\n");
    }
}

impl non_local::R for Local {
    async fn read<const L: usize>(&mut self) -> [u16; L] {
        //~^ ERROR: expected `impl Future<Output = [u16; L]>` to be a future that resolves to `[u8; L]`, but it resolves to `[u16; L]`
        [0; L]
    }
}
