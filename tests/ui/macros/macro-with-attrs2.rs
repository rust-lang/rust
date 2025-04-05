//@ run-pass

#[cfg(false)]
macro_rules! foo { () => (1) }

#[cfg(not(FALSE))]
macro_rules! foo { () => (2) }

pub fn main() {
    assert_eq!(foo!(), 2);
}
