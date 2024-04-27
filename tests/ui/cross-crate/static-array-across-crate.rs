//@ run-pass
#![allow(dead_code)]
//@ aux-build:pub_static_array.rs

extern crate pub_static_array as array;

use array::ARRAY;

static X: &'static u8 = &ARRAY[0];
static Y: &'static u8 = &(&ARRAY)[0];
static Z: u8 = (&ARRAY)[0];

pub fn main() {}
