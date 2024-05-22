// skip-filecheck
// Check that CopyProp does propagate return values of call terminators.
//@ test-mir-pass: CopyProp
//@ needs-unwind

#![feature(custom_mir, core_intrinsics)]
use std::intrinsics::mir::*;

#[inline(never)]
fn dummy(x: u8) -> u8 {
    x
}

// EMIT_MIR calls.nrvo.CopyProp.diff
fn nrvo() -> u8 {
    let y = dummy(5); // this should get NRVO
    y
}

// EMIT_MIR calls.multiple_edges.CopyProp.diff
#[custom_mir(dialect = "runtime", phase = "initial")]
fn multiple_edges(t: bool) -> u8 {
    mir! {
        let x: u8;
        {
            match t { true => bbt, _ => ret }
        }
        bbt = {
            Call(x = dummy(13), ReturnTo(ret), UnwindContinue())
        }
        ret = {
            // `x` is not assigned on the `bb0 -> ret` edge,
            // so should not be marked as SSA for merging with `_0`.
            RET = x;
            Return()
        }
    }
}

fn main() {
    // Make sure the function actually gets instantiated.
    nrvo();
    multiple_edges(false);
}
