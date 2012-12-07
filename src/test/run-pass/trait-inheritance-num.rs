use cmp::{Eq, Ord};
use num::from_int;

extern mod std;
use std::cmp::FuzzyEq;

pub trait NumExt: Num Eq Ord {}

pub trait FloatExt: NumExt FuzzyEq {}

fn greater_than_one<T:NumExt>(n: &T) -> bool { *n > from_int(1) }
fn greater_than_one_float<T:FloatExt>(n: &T) -> bool { *n > from_int(1) }

fn main() {}
