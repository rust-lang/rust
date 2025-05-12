//@ check-pass
#![allow(dead_code)]
struct S([u8; { struct Z; 0 }]);

fn main() {}
