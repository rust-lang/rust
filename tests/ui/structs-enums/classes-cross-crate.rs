//@ run-pass
//@ aux-build:cci_class_4.rs

extern crate cci_class_4;
use cci_class_4::kitties::cat;

pub fn main() {
    let mut nyan = cat(0_usize, 2, "nyan".to_string());
    nyan.eat();
    assert!(!nyan.eat());
    for _ in 1_usize..10_usize { nyan.speak(); };
    assert!(nyan.eat());
}
