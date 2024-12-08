//@ check-pass
#![allow(dead_code)]

fn broken(v: &[u8], i: usize, j: usize) -> &[u8] { &v[i..j] }

pub fn main() {}
