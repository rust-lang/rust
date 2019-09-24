// build-pass (FIXME(62277): could be check-pass?)

#![allow(dead_code)]
// pretty-expanded FIXME #23616

#![feature(non_ascii_idents)]
#![deny(non_snake_case)]

// This name is neither upper nor lower case
fn 你好() {}

fn main() {}
