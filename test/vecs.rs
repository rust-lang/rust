#![feature(custom_attribute)]
#![allow(dead_code, unused_attributes)]

#[miri_run]
fn make_vec() -> Vec<i32> {
    let mut v = Vec::with_capacity(4);
    v.push(1);
    v.push(2);
    v
}

#[miri_run]
fn make_vec_macro() -> Vec<i32> {
    vec![1, 2]
}
