//@ run-pass
use std::ops::{Add, Sub, Mul};

trait MyNum : Add<Output=Self> + Sub<Output=Self> + Mul<Output=Self> + PartialEq + Clone { }

#[derive(Clone, Debug)]
struct MyInt { val: isize }

impl Add for MyInt {
    type Output = MyInt;

    fn add(self, other: MyInt) -> MyInt { mi(self.val + other.val) }
}

impl Sub for MyInt {
    type Output = MyInt;

    fn sub(self, other: MyInt) -> MyInt { mi(self.val - other.val) }
}

impl Mul for MyInt {
    type Output = MyInt;

    fn mul(self, other: MyInt) -> MyInt { mi(self.val * other.val) }
}

impl PartialEq for MyInt {
    fn eq(&self, other: &MyInt) -> bool { self.val == other.val }
    fn ne(&self, other: &MyInt) -> bool { !self.eq(other) }
}

impl MyNum for MyInt {}

fn f<T:MyNum>(x: T, y: T) -> (T, T, T) {
    return (x.clone() + y.clone(), x.clone() - y.clone(), x * y);
}

fn mi(v: isize) -> MyInt { MyInt { val: v } }

pub fn main() {
    let (x, y) = (mi(3), mi(5));
    let (a, b, c) = f(x, y);
    assert_eq!(a, mi(8));
    assert_eq!(b, mi(-2));
    assert_eq!(c, mi(15));
}
