use std::cmp::PartialEq;
use std::ops::{Add, Sub, Mul};

pub trait MyNum : Add<Output=Self> + Sub<Output=Self> + Mul<Output=Self> + PartialEq + Clone {
}

#[derive(Clone, Debug)]
pub struct MyInt {
    pub val: isize
}

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

fn mi(v: isize) -> MyInt { MyInt { val: v } }
