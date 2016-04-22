#![feature(custom_attribute)]
#![allow(dead_code, unused_attributes)]

#[miri_run]
fn empty_array() -> [u16; 0] {
    []
}

#[miri_run]
fn big_array() -> [u16; 5] {
    [5, 4, 3, 2, 1]
}

#[miri_run]
fn array_array() -> [[u8; 2]; 3] {
    [[5, 4], [3, 2], [1, 0]]
}

#[miri_run]
fn index_unsafe() -> i32 {
    let a = [0, 10, 20, 30];
    unsafe { *a.get_unchecked(2) }
}

#[miri_run]
fn index() -> i32 {
    let a = [0, 10, 20, 30];
    a[2]
}

#[miri_run]
fn array_repeat() -> [u8; 8] {
    [42; 8]
}

fn main() {}
