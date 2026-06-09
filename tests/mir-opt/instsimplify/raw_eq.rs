//@ test-mir-pass: InstSimplify-after-simplifycfg
#![crate_type = "lib"]
#![feature(core_intrinsics)]
#![feature(custom_mir)]

// Custom MIR so we can get an argument that's not just a local directly
use std::intrinsics::mir::*;
use std::intrinsics::raw_eq;

// EMIT_MIR raw_eq.inner_array.InstSimplify-after-simplifycfg.diff
#[custom_mir(dialect = "runtime")]
pub fn inner_array(a: &&[i32; 2], b: &&[i32; 2]) -> bool {
    // CHECK-LABEL: fn inner_array(_1: &&[i32; 2], _2: &&[i32; 2]) -> bool
    // CHECK: [[AREF:_.+]] = copy (*_1);
    // CHECK: [[AINT:_.+]] = copy (*[[AREF]]) as u64 (Transmute);
    // CHECK: [[BREF:_.+]] = copy (*_2);
    // CHECK: [[BINT:_.+]] = copy (*[[BREF]]) as u64 (Transmute);
    // CHECK: _0 = Eq(move [[AINT]], move [[BINT]]);
    mir! {
        {
            Call(RET = raw_eq(*a, *b), ReturnTo(ret), UnwindUnreachable())
        }
        ret = {
            Return()
        }
    }
}
