//@ build-pass
//@ ignore-wasm32 no bare family
//@ ignore-sgx

#[cfg(windows)]
pub fn main() {
}

#[cfg(unix)]
pub fn main() {
}
