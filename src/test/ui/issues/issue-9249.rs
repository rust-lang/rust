// build-pass (FIXME(62277): could be check-pass?)
#![allow(dead_code)]
// pretty-expanded FIXME #23616

static DATA:&'static [&'static str] = &["my string"];
fn main() { }
