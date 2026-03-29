#![deny(dead_code)]
#![deny(unused_pub_items_in_binary)]

pub fn unused_pub_fn() {} //~ ERROR function `unused_pub_fn` is never used

pub fn used_pub_fn() {}

fn unused_priv_fn() {} //~ ERROR function `unused_priv_fn` is never used

fn main() {
    used_pub_fn();
}
