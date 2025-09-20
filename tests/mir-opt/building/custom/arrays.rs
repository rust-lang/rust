// skip-filecheck
#![feature(custom_mir, core_intrinsics)]

extern crate core;
use core::intrinsics::mir::*;

// EMIT_MIR arrays.arrays.built.after.mir
#[custom_mir(dialect = "built")]
fn arrays<const C: usize>() -> usize {
    mir! {
        {
            let x = [5_i32; C];
            let y = &raw const x;
            let z = CastUnsize::<_, *const [i32]>(y);
            let c = PtrMetadata(z);
            RET = c;
            Return()
        }
    }
}

fn main() {
    assert_eq!(arrays::<20>(), 20);
}
