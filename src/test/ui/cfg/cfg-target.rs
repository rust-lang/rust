// run-pass
#![feature(cfg_target)]

#[cfg(target = "x86_64-unknown-linux-gnu")]
pub fn main() {
}

#[cfg(not(target = "x86_64-unknown-linux-gnu"))]
pub fn main() {
}
