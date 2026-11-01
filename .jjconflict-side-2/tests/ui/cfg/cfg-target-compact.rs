//@ run-pass
#![feature(cfg_target_compact)]

#[cfg(target(os = "linux", pointer_width = "64"))]
pub fn main() {
}

#[cfg(not(target(os = "linux", pointer_width = "64")))]
pub fn main() {
}
