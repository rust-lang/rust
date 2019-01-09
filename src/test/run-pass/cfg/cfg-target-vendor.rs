// run-pass
#![feature(cfg_target_vendor)]

#[cfg(target_vendor = "unknown")]
pub fn main() {
}

#[cfg(not(target_vendor = "unknown"))]
pub fn main() {
}
