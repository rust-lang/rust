#![feature(custom_mir, core_intrinsics)]

extern crate core;
use core::intrinsics::mir::*;

#[repr(C)]
struct FieldStruct {
    first: u8,
    second: u16,
    third: u8
}

#[repr(C)]
struct NestedA {
    b: NestedB
}

#[repr(C)]
struct NestedB(u8);

// EMIT_MIR offset_of.offsets.built.after.mir
#[custom_mir(dialect = "analysis", phase = "post-cleanup")]
fn offsets() -> (usize, usize, usize, usize) {
    mir!(
        type RET = (usize, usize, usize, usize);
        {
            RET.0 = OffsetOf::<FieldStruct, 1>([0]);
            RET.1 = OffsetOf::<FieldStruct, 1>([1]);
            RET.2 = OffsetOf::<FieldStruct, 1>([2]);
            RET.3 = OffsetOf::<NestedA, 2>([0, 0]);
            Return()
        }
    )
}

fn main() {
    assert_eq!(offsets(), (0, 2, 4, 0));
}
