//@ check-pass
#![allow(dead_code)]
//@ pretty-expanded FIXME #23616

static DATA:&'static [&'static str] = &["my string"];
fn main() { }
