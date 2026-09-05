//@ test-mir-pass: MoveElimination
//@ compile-flags: -Cpanic=abort

#![feature(core_intrinsics, custom_mir)]
#![allow(dead_code)]
#![allow(internal_features)]

use std::intrinsics::mir::*;

#[derive(Copy, Clone)]
pub struct Triple(u8, u8, u8);

pub union U {
    a: [u8; 4],
    b: [u8; 4],
}

unsafe extern "C" {
    safe fn opaque_triple() -> Triple;
}

// EMIT_MIR alias_fixup.mixed_aggregate_aliasing.MoveElimination.diff
pub fn mixed_aggregate_aliasing(flag: bool, z: u8) -> Triple {
    // This checks an aggregate assignment on one branch after the other branch
    // remaps the input into the return place: overlapping field reads are
    // hoisted through temporaries before writing back into the return place.
    // CHECK-LABEL: fn mixed_aggregate_aliasing(
    // CHECK: debug z => _2;
    // CHECK: debug input => _0;
    // CHECK: debug out => _0;
    // CHECK: [[field1:_.*]] = copy (_0.1: u8);
    // CHECK: [[field0:_.*]] = no_retag move (_0.0: u8);
    // CHECK: (_0.0: u8) = no_retag move [[field1]];
    // CHECK: (_0.1: u8) = no_retag move [[field0]];
    // CHECK: (_0.2: u8) = no_retag move _2;
    let input = opaque_triple();
    let out = if flag { input } else { Triple(input.1, input.0, z) };
    out
}

// EMIT_MIR alias_fixup.aggregate_swap.MoveElimination.diff
pub fn aggregate_swap(x: u8, y: u8) -> (u8, u8) {
    // This checks that an aggregate swap-like assignment is safe after any
    // remapping that makes source fields share storage with destination fields.
    // CHECK-LABEL: fn aggregate_swap(
    // CHECK: debug pair => _0;
    // CHECK: [[saved:_.*]] = copy (_0.0: u8);
    // CHECK: [[tmp:_.*]] = no_retag move (_0.1: u8);
    // CHECK: (_0.0: u8) = no_retag move [[tmp]];
    // CHECK: (_0.1: u8) = no_retag move [[saved]];
    let mut pair = (x, y);
    let a = pair.0;
    let b = pair.1;
    pair = (b, a);
    pair
}

// EMIT_MIR alias_fixup.simple_partial_alias.MoveElimination.diff
pub fn simple_partial_alias(x: [u8; 4]) -> U {
    // This checks a non-aggregate assignment involving two same-typed union
    // fields, which are conservatively treated as aliasing.
    // CHECK-LABEL: fn simple_partial_alias(
    // CHECK: debug u => _0;
    // CHECK: _0 = U { a: move _1 };
    // CHECK: [[tmp:_.*]] = copy (_0.0: [u8; 4]);
    // CHECK: (_0.1: [u8; 4]) = move [[tmp]];
    let mut u = U { a: x };
    let tmp = unsafe { u.a };
    u.b = tmp;
    u
}

// EMIT_MIR alias_fixup.aggregate_indirect_source_alias.MoveElimination.diff
#[custom_mir(dialect = "runtime", phase = "post-cleanup")]
pub fn aggregate_indirect_source_alias() -> (u8, u8) {
    // This checks that, when an aggregate has a direct operand aliasing the
    // destination, indirect operands are also hoisted before field writes.
    // CHECK-LABEL: fn aggregate_indirect_source_alias(
    // CHECK: [[p:_.*]] = &raw const (_0.1: u8);
    // CHECK: [[tmp:_.*]] = no_retag copy (*[[p]]);
    // CHECK: (_0.0: u8) = no_retag move [[tmp]];
    mir! {
        let a: u8;
        let p: *const u8;
        let out: (u8, u8);

        {
            a = 1_u8;
            p = &raw const a;
            out = (*p, Move(a));
            RET = out;
            Return()
        }
    }
}
