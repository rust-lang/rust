// skip-filecheck
#![feature(custom_mir, core_intrinsics)]

extern crate core;
use core::intrinsics::mir::*;
use core::ptr::{addr_of, addr_of_mut};

// EMIT_MIR references.mut_ref.built.after.mir
#[custom_mir(dialect = "runtime", phase = "optimized")]
pub fn mut_ref(x: &mut i32) -> &mut i32 {
    mir! {
        let t: *mut i32;
        {
            t = addr_of_mut!(*x);
            RET = &mut *t;
            Retag(RET);
            Return()
        }
    }
}

// EMIT_MIR references.immut_ref.built.after.mir
#[custom_mir(dialect = "runtime", phase = "optimized")]
pub fn immut_ref(x: &i32) -> &i32 {
    mir! {
        let t: *const i32;
        {
            t = addr_of!(*x);
            RET = & *t;
            Retag(RET);
            Return()
        }
    }
}

// EMIT_MIR references.raw_pointer.built.after.mir
#[custom_mir(dialect = "built")]
pub fn raw_pointer(x: *const i32) -> *const i32 {
    // Regression test for a bug in which unsafetyck was not correctly turned off for
    // `dialect = "built"`
    mir! {
        {
            RET = addr_of!(*x);
            Return()
        }
    }
}

// EMIT_MIR references.raw_pointer_offset.built.after.mir
#[custom_mir(dialect = "built")]
pub fn raw_pointer_offset(x: *const i32) -> *const i32 {
    mir! {
        {
            RET = Offset(x, 1_isize);
            Return()
        }
    }
}

fn main() {
    let mut x = 5;
    let arr = [1, 2];
    assert_eq!(*mut_ref(&mut x), 5);
    assert_eq!(*immut_ref(&x), 5);
    unsafe {
        assert_eq!(*raw_pointer(addr_of!(x)), 5);
        assert_eq!(*raw_pointer_offset(addr_of!(arr[0])), 2);
    }
}
