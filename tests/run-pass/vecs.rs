#![feature(custom_attribute)]
#![allow(dead_code, unused_attributes)]

#[miri_run]
fn make_vec() -> Vec<u8> {
    let mut v = Vec::with_capacity(4);
    v.push(1);
    v.push(2);
    v
}

#[miri_run]
fn make_vec_macro() -> Vec<u8> {
    vec![1, 2]
}

#[miri_run]
fn make_vec_macro_repeat() -> Vec<u8> {
    vec![42; 5]
}

#[miri_run]
fn vec_into_iter() -> u8 {
    vec![1, 2, 3, 4]
        .into_iter()
        .map(|x| x * x)
        .fold(0, |x, y| x + y)
}

#[miri_run]
fn vec_reallocate() -> Vec<u8> {
    let mut v = vec![1, 2];
    v.push(3);
    v.push(4);
    v.push(5);
    v
}

#[miri_run]
fn main() {
    assert_eq!(vec_reallocate().len(), 5);
    assert_eq!(vec_into_iter(), 30);
    assert_eq!(make_vec().capacity(), 4);
}
