// build-pass (FIXME(62277): could be check-pass?)
#![allow(dead_code)]
struct S([u8; { struct Z; 0 }]);

fn main() {}
