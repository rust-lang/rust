//@ build-pass
//@ ignore-wasm32 no bare family
//@ ignore-sgx
//@ reference: cfg.target_family.unix
//@ reference: cfg.target_family.windows

#[cfg(windows)]
pub fn main() {
}

#[cfg(unix)]
pub fn main() {
}
