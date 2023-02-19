// unit-test: CopyProp

#![feature(custom_mir, core_intrinsics)]
#![allow(unused_assignments)]
extern crate core;
use core::intrinsics::mir::*;

fn opaque(_: impl Sized) -> bool { true }

fn cmp_ref(a: &u8, b: &u8) -> bool {
    std::ptr::eq(a as *const u8, b as *const u8)
}

#[custom_mir(dialect = "analysis", phase = "post-cleanup")]
fn f() -> bool {
    mir!(
        {
            let a = 5_u8;
            let r1 = &a;
            let b = a;
            // We cannot propagate the place `a`.
            let r2 = &b;
            Call(RET, next, cmp_ref(r1, r2))
        }
        next = {
            // But we can propagate the value `a`.
            Call(RET, ret, opaque(b))
        }
        ret = {
            Return()
        }
    )
}

fn main() {
    assert!(!f());
}

// EMIT_MIR borrowed_local.f.CopyProp.diff
