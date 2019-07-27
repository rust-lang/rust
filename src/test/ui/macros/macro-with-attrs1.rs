// run-pass
// compile-flags: --cfg foo


#[cfg(foo)]
macro_rules! foo { () => (1) }

#[cfg(not(foo))]
macro_rules! foo { () => (2) }

pub fn main() {
    assert_eq!(foo!(), 1);
}
