// skip-filecheck
#![feature(custom_mir, core_intrinsics)]

extern crate core;
use core::intrinsics::mir::*;

const D: i32 = 5;

// EMIT_MIR consts.consts.built.after.mir
#[custom_mir(dialect = "built")]
fn consts<const C: u32>() {
    mir! {
        {
            let _a = 5_u8;
            let _b = const { 5_i8 };
            let _c = C;
            let _d = D;
            let _e = consts::<10>;
            Return()
        }
    }
}

static S: i32 = 0x05050505;
static mut T: i32 = 0x0a0a0a0a;
// EMIT_MIR consts.statics.built.after.mir
#[custom_mir(dialect = "built")]
fn statics() {
    mir! {
        {
            let _a: &i32 = Static(S);
            let _b: *mut i32 = StaticMut(T);
            Return()
        }
    }
}

fn main() {
    consts::<5>();
    statics();
}
