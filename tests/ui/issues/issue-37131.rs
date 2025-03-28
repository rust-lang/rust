//~ ERROR can't find crate for `std`
// Tests that compiling for a target which is not installed will result in a helpful
// error message.

//@ compile-flags: --target=thumbv6m-none-eabi
//@ ignore-arm
//@ needs-llvm-components: arm

//@ error-pattern:target may not be installed
fn main() { }
