// build-pass
// ignore-sgx

// pretty-expanded FIXME #23616

#[cfg(target_family = "windows")]
pub fn main() {}

#[cfg(target_family = "unix")]
pub fn main() {}

#[cfg(all(target_family = "wasm", not(target_os = "emscripten")))]
pub fn main() {}
