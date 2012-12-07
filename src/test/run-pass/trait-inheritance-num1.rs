// Using the real Num from core

use cmp::Ord;
use num::from_int;

pub trait NumExt: Num Ord { }

fn greater_than_one<T:NumExt>(n: &T) -> bool {
    *n > from_int(1)
}

fn main() {}
