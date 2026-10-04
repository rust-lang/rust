//@ run-pass
#![allow(dead_code)]
// Extending Num and using inherited static methods


pub trait NumCast: Sized {
    fn from(i: i32) -> Option<Self>;
}

pub trait Num {
    fn from_int(i: isize) -> Self;
    fn gt(&self, other: &Self) -> bool;
}

pub trait NumExt: NumCast + PartialOrd { }

fn greater_than_one<T:NumExt>(n: &T) -> bool {
    n.gt(&NumCast::from(1).unwrap())
}

pub fn main() {}
