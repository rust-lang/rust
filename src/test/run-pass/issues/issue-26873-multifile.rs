// run-pass
#![allow(dead_code)]
#![allow(unused_imports)]
#![allow(non_snake_case)]

// ignore-pretty issue #37195

#[path = "issue-26873-multifile/mod.rs"]
mod multifile;

fn main() {}
