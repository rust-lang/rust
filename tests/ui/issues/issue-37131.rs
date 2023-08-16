// Tests that compiling for a target which is not installed will result in a helpful
// error message.

//@compile-flags: --target=thumbv6m-none-eabi
//@ignore-target-arm
// needs-llvm-components: arm

//@error-in-other-file:target may not be installed
fn main() { }
