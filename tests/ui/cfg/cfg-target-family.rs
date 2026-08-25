//@ build-pass
//@ ignore-sgx
//@ reference: cfg.target_family.def
//@ reference: cfg.target_family.values


#[cfg(target_family = "windows")]
pub fn main() {}

#[cfg(target_family = "unix")]
pub fn main() {}

#[cfg(all(target_family = "wasm", not(target_os = "emscripten")))]
pub fn main() {}
