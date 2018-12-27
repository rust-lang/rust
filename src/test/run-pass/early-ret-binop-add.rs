#![allow(dead_code)]
#![allow(unreachable_code)]
// pretty-expanded FIXME #23616

use std::ops::Add;

fn wsucc<T:Add<Output=T> + Copy>(n: T) -> T { n + { return n } }

pub fn main() { }
