#![deny(dead_code)]
#![allow(dead_code_pub_in_binary)]

pub fn unused_pub_fn() {
    helper();
}

fn helper() {}

fn unused_priv_fn() {} //~ ERROR function `unused_priv_fn` is never used

fn main() {}
