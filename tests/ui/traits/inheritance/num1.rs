//@ run-pass
#![allow(dead_code)]

pub trait NumCast: Sized {
    fn from(i: i32) -> Option<Self>;
}

pub trait NumExt: NumCast + PartialOrd { }

fn greater_than_one<T:NumExt>(n: &T) -> bool {
    *n > NumCast::from(1).unwrap()
}

pub fn main() {}
