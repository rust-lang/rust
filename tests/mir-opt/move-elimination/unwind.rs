//@ test-mir-pass: MoveElimination
//@ compile-flags: -Cpanic=unwind

#![crate_type = "lib"]
#![feature(custom_mir, core_intrinsics)]
use std::intrinsics::mir::*;

#[inline(never)]
fn opaque() -> u32 {
    1
}

// EMIT_MIR unwind.call_destination.MoveElimination.diff
#[custom_mir(dialect = "runtime", phase = "post-cleanup")]
pub fn call_destination() {
    // CHECK-LABEL: fn call_destination(
    // CHECK: [[DEST:_[0-9]+]] = opaque()
    // CHECK: [[SRC:_[0-9]+]] = const 2_u32;
    // CHECK-NEXT: [[DEST]] = move [[SRC]];
    // Unwinding doesn't deallocate the call destination, it only leaves it
    // uninitialized. This checks that dest is not merged with src.
    mir! {
        let dest: u32;
        let src: u32;
        let p: *const u32;
        {
            dest = 0;
            p = &raw const dest;
            Call(dest = opaque(), ReturnTo(done), UnwindCleanup(cleanup))
        }
        done = {
            RET = ();
            Return()
        }
        cleanup (cleanup) = {
            src = 2;
            dest = Move(src);
            UnwindResume()
        }
    }
}
