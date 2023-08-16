//@run
//@aux-build:anon-extern-mod-cross-crate-1.rs
//@aux-build:anon-extern-mod-cross-crate-1.rs
// pretty-expanded FIXME #23616
//@ignore-target-wasm32-unknown-unknown no libc to test ffi with

extern crate anonexternmod;

pub fn main() {}
