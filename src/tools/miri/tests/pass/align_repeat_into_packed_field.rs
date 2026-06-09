#![feature(custom_mir, core_intrinsics)]
use std::intrinsics::mir::*;

#[repr(packed)]
struct S {
    field: [u32; 2],
}

#[custom_mir(dialect = "runtime", phase = "optimized")]
fn test() {
    mir! {
        let s: S;
        {
            // Store a repeat expression directly into a field of a packed struct.
            s.field = [0; 2];
            Return()
        }
    }
}

fn main() {
    // Run this a bunch of time to make sure it doesn't pass by chance.
    for _ in 0..20 {
        test();
    }
}
