//@ compile-flags: -Ztreat-pub-as-pub-crate
#![feature(rustc_attrs)]
#![deny(dead_code)]

pub fn unused_pub_fn() {} //~ ERROR function `unused_pub_fn` is never used

pub fn used_pub_fn() {}

fn main() {
    used_pub_fn();
}
