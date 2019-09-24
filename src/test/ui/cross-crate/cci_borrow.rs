// run-pass
// aux-build:cci_borrow_lib.rs

#![feature(box_syntax)]

extern crate cci_borrow_lib;
use cci_borrow_lib::foo;

pub fn main() {
    let p: Box<_> = box 22;
    let r = foo(&*p);
    println!("r={}", r);
    assert_eq!(r, 22);
}
