// aux-build:attr-cfg.rs
// revisions: foo bar

extern crate attr_cfg;
use attr_cfg::attr_cfg;

#[attr_cfg]
fn outer() -> u8 {
    #[cfg(foo)]
    fn inner() -> u8 { 1 }

    #[cfg(bar)]
    fn inner() -> u8 { 2 }

    inner()
}

#[cfg(foo)]
fn main() {
    assert_eq!(outer(), 1);
}

#[cfg(bar)]
fn main() {
    assert_eq!(outer(), 2);
}
