// run-pass
#![allow(dead_code)]

#![feature(const_let)]

type Array = [u32; {  let x = 2; 5 }];

pub fn main() {}
