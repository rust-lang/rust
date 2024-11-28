//@ run-pass

#![allow(dead_code)]
#![allow(unreachable_code)]

use std::ops::Add;

fn wsucc<T:Add<Output=T> + Copy>(n: T) -> T { n + { return n } }

pub fn main() { }
