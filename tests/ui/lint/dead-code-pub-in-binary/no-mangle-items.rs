#![deny(dead_code)]
#![deny(dead_code_pub_in_binary)]

#[no_mangle]
pub fn pub_fn_no_mangle() {}

pub fn unused_pub_fn() {} //~ ERROR function `unused_pub_fn` is never used

#[no_mangle]
pub fn unused_priv_fn_no_mangle() {}

fn unused_priv_fn() {} //~ ERROR function `unused_priv_fn` is never used

fn main() {}
