#![allow(dead_code)]
// Test several functions can be used for constants
// 1. Vec::new()
// 2. String::new()

#![feature(const_vec_new)]
#![feature(const_string_new)]

const MY_VEC: Vec<usize> = Vec::new();

const MY_STRING: String = String::new();

pub fn main() {}
