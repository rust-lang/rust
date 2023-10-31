// unit-test: CopyProp
//
// This attempts to mutate `a` via a pointer derived from `addr_of!(a)`. That is UB
// according to Miri. However, the decision to make this UB - and to allow
// rustc to rely on that fact for the purpose of optimizations - has not been
// finalized.
//
// As such, we include this test to ensure that copy prop does not rely on that
// fact. Specifically, if `addr_of!(a)` could not be used to modify a, it would
// be correct for CopyProp to replace all occurrences of `a` with `c` - but that
// would cause `f(true)` to output `false` instead of `true`.

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
