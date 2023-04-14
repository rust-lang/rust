// run-pass
#![allow(unused_assignments)]

pub fn main() {
    let i: Box<_> = Box::new(1);
    let mut j: Box<_> = Box::new(2);
    // Should drop the previous value of j
    j = i;
    assert_eq!(*j, 1);
}
