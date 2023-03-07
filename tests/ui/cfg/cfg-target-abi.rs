// run-pass
#![feature(cfg_target_abi)]

#[cfg(target_abi = "eabihf")]
pub fn main() {
}

#[cfg(not(target_abi = "eabihf"))]
pub fn main() {
}
