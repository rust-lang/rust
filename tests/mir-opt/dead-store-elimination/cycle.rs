// This example is interesting because the non-transitive version of `MaybeLiveLocals` would
// report that *all* of these stores are live.
//
//@ needs-unwind
//@ test-mir-pass: DeadStoreElimination-initial

#![feature(core_intrinsics, custom_mir)]
use std::intrinsics::mir::*;

#[inline(never)]
fn cond() -> bool {
    false
}

// EMIT_MIR cycle.cycle.DeadStoreElimination-initial.diff
#[custom_mir(dialect = "runtime", phase = "post-cleanup")]
fn cycle(mut x: i32, mut y: i32, mut z: i32) {
    // CHECK-LABEL: fn cycle(
    // CHECK-NOT: {{_.*}} = {{_.*}};
    // CHECK-NOT: {{_.*}} = move {{_.*}};

    // We use custom MIR to avoid generating debuginfo, that would force to preserve writes.
    mir! {
        let condition: bool;
        {
            Call(condition = cond(), ReturnTo(bb1), UnwindContinue())
        }
        bb1 = {
            match condition { true => bb2, _ => ret }
        }
        bb2 = {
            let temp = z;
            z = y;
            y = x;
            x = temp;
            Call(condition = cond(), ReturnTo(bb1), UnwindContinue())
        }
        ret = {
            Return()
        }
    }
}

fn main() {
    cycle(1, 2, 3);
}
