//@ aux-build:extern-statics.rs

extern crate extern_statics;
use extern_statics::*;

extern "C" {
    static mut B: u8;
}

fn main() {
    let b = B; //~ static is unsafe
    let rb = &B; //~ static is unsafe
                 //~^ WARN shared reference to mutable static is discouraged [static_mut_refs]
    let xb = XB; //~ static is unsafe
    let xrb = &XB; //~ static is unsafe
                   //~^ WARN shared reference to mutable static is discouraged [static_mut_refs]
}
