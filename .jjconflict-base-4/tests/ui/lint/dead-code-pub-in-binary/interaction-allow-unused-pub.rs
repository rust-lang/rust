#![allow(dead_code_pub_in_binary)]
#![deny(dead_code)]

pub fn unused_pub_fn() {}
fn unused_priv_fn() {} //~ ERROR function `unused_priv_fn` is never used

fn main() {}
