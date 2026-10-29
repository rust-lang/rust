//@ run-pass
#![allow(dead_code)]
#![allow(unused_imports)]
#![allow(non_snake_case)]


#[path = "issue-26873-multifile/mod.rs"]
mod multifile;

fn main() {}
