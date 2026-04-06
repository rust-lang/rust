#![deny(dead_code)]
#![deny(unused_pub_items_in_binary)]
#![crate_type = "lib"]

pub fn unused_pub_fn() {} // Should NOT error because it's a library crate

fn unused_priv_fn() {} //~ ERROR function `unused_priv_fn` is never used
