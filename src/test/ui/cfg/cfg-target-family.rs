// run-pass
// ignore-cloudabi no target_family
// ignore-wasm32-bare no target_family
// ignore-sgx

// pretty-expanded FIXME #23616

#[cfg(target_family = "windows")]
pub fn main() {
}

#[cfg(target_family = "unix")]
pub fn main() {
}
