//@ build-pass
//@ reference: cfg.target_abi.def
//@ reference: cfg.target_abi.values

#[cfg(target_abi = "eabihf")]
pub fn main() {
}

#[cfg(not(target_abi = "eabihf"))]
pub fn main() {
}
