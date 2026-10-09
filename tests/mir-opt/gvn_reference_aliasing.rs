//@ test-mir-pass: GVN
//@ compile-flags: -O
// EMIT_MIR_FOR_EACH_PANIC_STRATEGY

// This is a regression test for github.com/rust-lang/rust/issues/163825.
// The goal is to check that for the read-write-read pattern, GVN does not unify the
// two reads, and the second reads the new value written through the pointer.

#![crate_type = "lib"]

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
