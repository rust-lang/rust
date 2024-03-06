//@ run-pass

#[cfg(target_abi = "eabihf")]
pub fn main() {
}

#[cfg(not(target_abi = "eabihf"))]
pub fn main() {
}
