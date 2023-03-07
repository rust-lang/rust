// run-pass
// aux-build:cci_borrow_lib.rs

extern crate cci_borrow_lib;
use cci_borrow_lib::foo;

pub fn main() {
    let p: Box<_> = Box::new(22);
    let r = foo(&*p);
    println!("r={}", r);
    assert_eq!(r, 22);
}
