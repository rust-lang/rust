//@ test-mir-pass: GVN

#![feature(custom_mir, core_intrinsics)]
extern crate core;
use core::intrinsics::mir::*;

#[custom_mir(dialect = "runtime", phase = "initial")]
fn fn0() {
    // CHECK-LABEL: fn fn0(
    mir! {
        let a: usize;
        let b: [u128; 6];
        let c: ([u128; 6],);
        let d: ([u128; 6],);
        let x: ();
        {
            // CHECK: bb0: {
            // CHECK-NEXT: _1 = const 1_usize;
            // CHECK-NEXT: _2 = [const 42_u128; 6];
            // CHECK-NEXT: _2[1 of 2] = const 1_u128;
            // CHECK-NEXT: _3 = (copy _2,);
            // CHECK-NEXT: _4 = copy _3;
            // CHECK-NEXT: _5 = fn1(copy (_3.0: [u128; 6]), copy _3)
            a = 1_usize;
            b = [42; 6];
            b[a] = 1;
            c = (b,);
            d = c;
            Call(x = fn1(Move(c.0), d), ReturnTo(bb1), UnwindUnreachable())
        }
        bb1 = {
            Return()
        }
    }
}

#[inline(never)]
fn fn1(a: [u128; 6], mut b: ([u128; 6],)) {
    b.0 = [0; 6];
}

fn main() {
    fn0();
}

// EMIT_MIR gvn_copy_moves.fn0.GVN.diff
