// build-pass (FIXME(62277): could be check-pass?)
#![allow(dead_code)]
// pretty-expanded FIXME #23616

fn broken(v: &[u8], i: usize, j: usize) -> &[u8] { &v[i..j] }

pub fn main() {}
