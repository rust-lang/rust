// skip-filecheck
#![feature(custom_mir, core_intrinsics, inline_const)]

extern crate core;
use core::intrinsics::mir::*;

// EMIT_MIR arrays.arrays.built.after.mir
#[custom_mir(dialect = "built")]
fn arrays<const C: usize>() -> usize {
    mir!({
        let x = [5_i32; C];
        let c = Len(x);
        RET = c;
        Return()
    })
}

fn main() {
    assert_eq!(arrays::<20>(), 20);
}
