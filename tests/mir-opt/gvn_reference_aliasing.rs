//@ test-mir-pass: GVN
//@ compile-flags: -O -Zinline-mir
// For the ManuallyDrop case we need to inline its Deref impl to expose the same MIR pattern caused
// by the interior MaybeDangling.

// This is a regression test for github.com/rust-lang/rust/issues/163825.
// The goal is to check that for the read-write-read pattern, GVN does not unify the
// two reads, and the second reads the new value written through the pointer.

#![crate_type = "lib"]

use std::mem::ManuallyDrop;

union U {
    r: &'static u8,
    p: *mut u8,
}

// EMIT_MIR gvn_reference_aliasing.union_field.GVN.diff
#[inline(never)]
pub fn union_field(p: *mut u8) -> (u8, u8) {
    // CHECK-LABEL: fn union_field(
    // CHECK: [[UNION:_.*]] = U { r: copy _1 };
    // CHECK: [[REF:_.*]] = copy ([[UNION]].0: &u8);
    // CHECK: [[VAL:_.*]] = copy (*[[REF]]);
    // CHECK: (*_1) = const 2_u8;
    // CHECK: [[REF2:_.*]] = copy ([[UNION]].0: &u8);
    // CHECK: [[VAL2:_.*]] = copy (*[[REF2]]);
    // CHECK: _0 = (copy [[VAL]], copy [[VAL2]]);
    let u = U { p };
    unsafe {
        let a = u.r;
        let a = *a;
        *p = 2;
        let b = u.r;
        let b = *b;
        (a, b)
    }
}

// EMIT_MIR gvn_reference_aliasing.manuallydrop.GVN.diff
#[inline(never)]
pub fn manuallydrop(p: *mut u8) -> (u8, u8) {
    // CHECK-LABEL: fn manuallydrop(
    // CHECK: [[MD:_.*]] = copy _1 as std::mem::ManuallyDrop<&u8> (Transmute);
    // CHECK: [[MDREF:_.*]] = &[[MD]];
    // CHECK: [[FIELD:_.*]] = &((*[[MDREF]]).0: &u8);
    // CHECK: [[REF:_.*]] = copy (*[[FIELD]]);
    // CHECK: [[VAL:_.*]] = copy (*[[REF]]);
    // CHECK: (*_1) = const 2_u8;
    // CHECK: [[MDREF2:_.*]] = &[[MD]];
    // CHECK: [[FIELD2:_.*]] = &((*[[MDREF2]]).0: &u8);
    // CHECK: [[REF2:_.*]] = copy (*[[FIELD2]]);
    // CHECK: [[VAL2:_.*]] = copy (*[[REF2]]);
    // CHECK: _0 = (copy [[VAL]], copy [[VAL2]]);
    let m: ManuallyDrop<&u8> = unsafe { std::mem::transmute(p) };
    unsafe {
        let a = *m;
        let a = *a;
        *p = 2;
        let b = *m;
        let b = *b;
        (a, b)
    }
}
