// run-pass
#![feature(fn_traits)]

use std::ops::Fn;

fn say(x: u32, y: u32) {
    println!("{} {}", x, y);
}

fn main() {
    Fn::call(&say, (1, 2));
}
