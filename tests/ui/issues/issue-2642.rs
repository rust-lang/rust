//@ run-pass
#![allow(dead_code)]

fn f() {
   let _x: usize = loop { loop { break; } };
}

pub fn main() {
}
