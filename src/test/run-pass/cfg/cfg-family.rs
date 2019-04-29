// run-pass
// pretty-expanded FIXME #23616

#[cfg(windows)]
pub fn main() {
}

#[cfg(unix)]
pub fn main() {
}

#[cfg(not(any(windows, unix)))]
pub fn main() {
}
