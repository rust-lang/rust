//~ ERROR can't find crate for `core`
//~^ ERROR can't find crate for `compiler_builtins`

// compile-flags: --target thumbv7em-none-eabihf
// needs-llvm-components: arm
#![deny(unsafe_code)]
#![deny(warnings)]
#![no_std]

extern crate cortex_m;
//~^ ERROR can't find crate for `cortex_m`

fn main() {}
