//@build-pass
//@compile-flags: -Clink-dead-code=true --crate-type lib
// We used to not handle all "rustic" ABIs in a (relatively) uniform way,
// so we failed to fix up arguments for actually passing through the ABI...
#![feature(rust_cold_cc)]
pub extern "rust-cold" fn foo(_: [usize; 3]) {}
