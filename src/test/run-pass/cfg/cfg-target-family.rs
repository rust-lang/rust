// run-pass

// pretty-expanded FIXME #23616

#[cfg(target_family = "windows")]
pub fn main() {
}

#[cfg(target_family = "unix")]
pub fn main() {
}

#[cfg(not(any(target_family = "windows", target_family = "unix")))]
pub fn main() {
}