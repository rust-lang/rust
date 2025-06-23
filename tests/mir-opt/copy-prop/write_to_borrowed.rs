//@ test-mir-pass: CopyProp

#![feature(custom_mir, core_intrinsics)]
#![allow(internal_features)]

use std::intrinsics::mir::*;

#[custom_mir(dialect = "runtime")]
fn main() {
    mir! {
        // Both _3 and _5 are borrowed, check that we do not unify them, and that we do not
        // introduce a write to any of them.
        let _1;
        let _2;
        let _3;
        let _4;
        let _5;
        let _6;
        let _7;
        // CHECK: bb0: {
        {
            // CHECK-NEXT: _1 = &raw const _2;
            _1 = core::ptr::addr_of!(_2);
            // CHECK-NEXT: _3 = const 'b';
            _3 = 'b';
            // CHECK-NEXT: _5 = copy _3;
            _5 = _3;
            // CHECK-NEXT: _6 = &_3;
            _6 = &_3;
            // CHECK-NOT: {{_.*}} = {{_.*}};
            _4 = _5;
            // CHECK-NEXT: (*_1) = copy (*_6);
            *_1 = *_6;
            // CHECK-NEXT: _6 = &_5;
            _6 = &_5;
            // CHECK-NEXT: _7 = dump_var::<char>(copy _5)
            Call(_7 = dump_var(_4), ReturnTo(bb1), UnwindUnreachable())
        }
        bb1 = { Return() }
    }
}

fn dump_var<T>(_: T) {}

// EMIT_MIR write_to_borrowed.main.CopyProp.diff
