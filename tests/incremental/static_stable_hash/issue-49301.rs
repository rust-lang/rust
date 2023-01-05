// https://github.com/rust-lang/rust/issues/49081

// revisions:rpass1 rpass2

#[cfg(rpass1)]
pub static A: &str = "hello";
#[cfg(rpass2)]
pub static A: &str = "xxxxx";

#[cfg(rpass1)]
fn main() {
    assert_eq!(A, "hello");
}

#[cfg(rpass2)]
fn main() {
    assert_eq!(A, "xxxxx");
}
