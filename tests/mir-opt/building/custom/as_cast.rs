#![feature(custom_mir, core_intrinsics)]

extern crate core;
use core::intrinsics::mir::*;

// EMIT_MIR as_cast.int_to_int.built.after.mir
#[custom_mir(dialect = "built")]
fn int_to_int(x: u32) -> i32 {
    mir!(
        {
            RET = x as i32;
            Return()
        }
    )
}

// EMIT_MIR as_cast.float_to_int.built.after.mir
#[custom_mir(dialect = "built")]
fn float_to_int(x: f32) -> i32 {
    mir!(
        {
            RET = x as i32;
            Return()
        }
    )
}

// EMIT_MIR as_cast.int_to_ptr.built.after.mir
#[custom_mir(dialect = "built")]
fn int_to_ptr(x: usize) -> *const i32 {
    mir!(
        {
            RET = x as *const i32;
            Return()
        }
    )
}

fn main() {
    assert_eq!(int_to_int(5), 5);
    assert_eq!(float_to_int(5.), 5);
    assert_eq!(int_to_ptr(0), std::ptr::null());
}
