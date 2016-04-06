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

#[miri_run]
fn make_vec_macro_repeat() -> Vec<u8> {
    vec![42; 5]
}

#[miri_run]
fn vec_into_iter() -> i32 {
    vec![1, 2, 3, 4].into_iter().fold(0, |x, y| x + y)
}
