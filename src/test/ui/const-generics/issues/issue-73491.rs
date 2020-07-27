// check-pass
#![feature(const_generics)]
#![allow(incomplete_features)]

const LEN: usize = 1024;

fn hoge<const IN: [u32; LEN]>() {}

fn main() {}
