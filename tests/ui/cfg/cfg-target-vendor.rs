// run-pass
#[cfg(target_vendor = "unknown")]
pub fn main() {
}

#[cfg(not(target_vendor = "unknown"))]
pub fn main() {
}
