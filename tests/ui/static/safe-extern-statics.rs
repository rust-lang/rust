// aux-build:extern-statics.rs
// revisions: mir thir
// [thir]compile-flags: -Z thir-unsafeck

extern crate extern_statics;
use extern_statics::*;

extern "C" {
    static A: u8;
}

fn main() {
    let a = A; //~ ERROR use of extern static is unsafe
    let ra = &A; //~ ERROR use of extern static is unsafe
    let xa = XA; //~ ERROR use of extern static is unsafe
    let xra = &XA; //~ ERROR use of extern static is unsafe
}
