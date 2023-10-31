// check-pass
#![allow(improper_ctypes_definitions)]
#![crate_type = "lib"]

// Check that computing the fn abi for `bad`, with a external ABI fn ptr that is not FFI-safe, does
// not ICE.

pub fn bad(f: extern "C" fn([u8])) {}
