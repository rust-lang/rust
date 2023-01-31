#![feature(custom_mir, core_intrinsics)]
#![allow(unused_assignments)]
extern crate core;
use core::intrinsics::mir::*;

#[custom_mir(dialect = "analysis", phase = "post-cleanup")]
fn f(c: bool) -> bool {
    mir!({
        let a = c;
        let p = core::ptr::addr_of!(a);
        let p2 = core::ptr::addr_of_mut!(*p);
        *p2 = false;
        RET = c;
        Return()
    })
}

fn main() {
    assert_eq!(true, f(true));
}

// EMIT_MIR mutate_through_pointer.f.CopyProp.diff
