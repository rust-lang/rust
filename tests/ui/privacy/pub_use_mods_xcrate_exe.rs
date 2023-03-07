// run-pass
// aux-build:pub_use_mods_xcrate.rs

// pretty-expanded FIXME #23616

#![allow(unused_imports)]

extern crate pub_use_mods_xcrate;
use pub_use_mods_xcrate::a::c;

pub fn main(){}
