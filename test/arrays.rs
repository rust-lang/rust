#![feature(custom_attribute)]
#![allow(dead_code, unused_attributes)]

#[miri_run]
fn empty_array() -> [u16; 0] {
    []
}

#[miri_run]
fn singular_array() -> [u16; 1] {
    [42]
}

#[miri_run]
fn deuce_array() -> [u16; 2] {
    [42, 53]
}

#[miri_run]
fn big_array() -> [u16; 5] {
    [5, 4, 3, 2, 1]
}

#[miri_run]
fn array_array() -> [[u8; 2]; 3] {
    [[5, 4], [3, 2], [1, 0]]
}
