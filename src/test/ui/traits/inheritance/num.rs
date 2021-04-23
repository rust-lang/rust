// run-pass
#![allow(dead_code)]
#![allow(unused_variables)]
// pretty-expanded FIXME #23616

pub trait NumExt: PartialEq + PartialOrd {}

pub trait FloatExt: NumExt {}

fn greater_than_one<T: NumExt>(n: &T) -> bool { loop {} }
fn greater_than_one_float<T: FloatExt>(n: &T) -> bool { loop {} }

pub fn main() {}
