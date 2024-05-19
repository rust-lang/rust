//@ run-pass
#![allow(unused_allocation)]

pub fn main() {
    let i: Box<_> = Box::new(100);
    assert_eq!(i, Box::new(100));
    assert!(i < Box::new(101));
    assert!(i <= Box::new(100));
    assert!(i > Box::new(99));
    assert!(i >= Box::new(99));
}
