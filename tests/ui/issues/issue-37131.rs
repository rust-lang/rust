//~ ERROR can't find crate for `std`
//~| NOTE target may not be installed
//~| NOTE can't find crate
// Tests that compiling for a target which is not installed will result in a helpful
// error message.

//@ compile-flags: --target=thumbv6m-none-eabi
//@ ignore-arm
//@ needs-llvm-components: arm

fn main() { }
