// run-pass

#![allow(dead_code)]
#![allow(unreachable_code)]

fn dont_call_me() { panic!(); println!("{}", 1); }

pub fn main() { }
