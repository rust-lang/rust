//@ run-pass
#![allow(dead_code)]

trait MyNum : PartialEq { }

#[derive(Debug)]
struct MyInt { val: isize }

impl PartialEq for MyInt {
    fn eq(&self, other: &MyInt) -> bool { self.val == other.val }
    fn ne(&self, other: &MyInt) -> bool { !self.eq(other) }
}

impl MyNum for MyInt {}

fn f<T:MyNum>(x: T, y: T) -> bool {
    return x == y;
}

fn mi(v: isize) -> MyInt { MyInt { val: v } }

pub fn main() {
    let (x, y, z) = (mi(3), mi(5), mi(3));
    assert!(x != y);
    assert_eq!(x, z);
}
