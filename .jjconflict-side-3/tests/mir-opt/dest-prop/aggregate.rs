//@ test-mir-pass: DestinationPropagation
// EMIT_MIR_FOR_EACH_PANIC_STRATEGY

#![feature(custom_mir, core_intrinsics)]
#![allow(internal_features)]

use std::intrinsics::mir::*;
use std::mem::MaybeUninit;

fn dump_var<T>(_: T) {}

// EMIT_MIR aggregate.rewrap.DestinationPropagation.diff
#[custom_mir(dialect = "runtime")]
fn rewrap() -> (u8,) {
    // CHECK-LABEL: fn rewrap(
    // CHECK: (_0.0: u8) = const 0_u8;
    // CHECK: _2 = (copy (_0.0: u8),);
    // CHECK: _0 = copy _2;
    mir! {
        let _1: (u8,);
        let _2: (u8,);
        {
            _1.0 = 0;
            RET = _1;
            _2 = (RET.0, );
            RET = _2;
            Return()
        }
    }
}

// EMIT_MIR aggregate.swap.DestinationPropagation.diff
#[custom_mir(dialect = "runtime")]
fn swap() -> (MaybeUninit<[u8; 10]>, MaybeUninit<[u8; 10]>) {
    // CHECK-LABEL: fn swap(
    // CHECK: _0 = const
    // CHECK: _2 = copy _0;
    // CHECK: _0 = (copy (_2.1: {{.*}}), copy (_2.0: {{.*}}));
    mir! {
        let _1: (MaybeUninit<[u8; 10]>, MaybeUninit<[u8; 10]>);
        let _2: (MaybeUninit<[u8; 10]>, MaybeUninit<[u8; 10]>);
        let _3: ();
        {
            _1 = const { (MaybeUninit::new([0; 10]), MaybeUninit::new([1; 10])) };
            _2 = _1;
            _1 = (_2.1, _2.0);
            RET = _1;
            Return()
        }
    }
}
