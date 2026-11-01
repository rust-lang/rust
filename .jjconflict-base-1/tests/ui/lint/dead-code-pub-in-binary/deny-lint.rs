#![deny(dead_code_pub_in_binary)]
#![allow(dead_code)]

pub fn unused_pub_fn() {} //~ ERROR function `unused_pub_fn` is never used

fn main() {}
