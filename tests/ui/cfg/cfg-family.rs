// build-pass
// pretty-expanded FIXME #23616
//@ignore-target-wasm32-unknown-unknown no bare family
//@ignore-target-sgx

#[cfg(windows)]
pub fn main() {}

#[cfg(unix)]
pub fn main() {}
