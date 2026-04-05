#![allow(dead_code)]
#![deny(unused_pub_items_in_binary)]

pub fn unused_pub_fn() {} //~ ERROR function `unused_pub_fn` is never used
fn unused_priv_fn() {}

fn main() {}
