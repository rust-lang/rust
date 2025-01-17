//@ run-pass
#![allow(dead_code)]
// Test that wrapping an unsized struct in an enum which gets optimised does
// not ICE.


struct Str {
    f: [u8]
}

fn main() {
    let str: Option<&Str> = None;
    let _ = str.is_some();
}
