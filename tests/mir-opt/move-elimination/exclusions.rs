//@ test-mir-pass: MoveElimination
//@ compile-flags: -Cpanic=abort -Zmir-enable-passes=+DeadStoreElimination-initial

#![feature(repr_simd)]

pub struct Fields {
    data: [u8; 8],
    tag: u8,
}

#[repr(packed)]
struct Packed {
    a: [u8; 8],
    b: [u8; 8],
}

#[repr(simd)]
struct U32x4([u32; 4]);

unsafe extern "C" {
    safe fn observe(_: *const Fields);
    safe fn make_fields(_: u8) -> Fields;
}

// EMIT_MIR exclusions.index_local_not_projected.MoveElimination.diff
pub fn index_local_not_projected(a: [usize; 4], i: usize) -> [usize; 1] {
    // This checks that a local used as an array index is kept as a bare local,
    // because it cannot later be rewritten to a projection like `_0[0]`.
    // CHECK-LABEL: fn index_local_not_projected(
    // CHECK: debug idx => _2;
    // CHECK: _4 = copy _2;
    // CHECK: _0 = [move _2];
    let idx = i;
    let _ = a[idx];
    [idx]
}

// EMIT_MIR exclusions.packed_fields_not_projected.MoveElimination.diff
pub fn packed_fields_not_projected() -> Packed {
    // This checks that aggregate fields are not remapped into packed struct
    // fields, which could create unaligned projected places.
    // CHECK-LABEL: fn packed_fields_not_projected(
    // CHECK: debug a => [[a:_.*]];
    // CHECK: debug b => [[b:_.*]];
    // CHECK: _0 = Packed { a: move [[a]], b: move [[b]] };
    let a = [1; 8];
    let b = [2; 8];
    Packed { a, b }
}

// EMIT_MIR exclusions.simd_field_not_projected.MoveElimination.diff
pub fn simd_field_not_projected() -> U32x4 {
    // This checks that aggregate fields are not remapped into repr(simd) ADTs,
    // since optimized MIR must not project into SIMD vectors.
    // CHECK-LABEL: fn simd_field_not_projected(
    // CHECK: debug lanes => [[lanes:_.*]];
    // CHECK: _0 = U32x4(move [[lanes]]);
    let lanes = [1, 2, 3, 4];
    U32x4(lanes)
}

// EMIT_MIR exclusions.overlapping_lifetimes.MoveElimination.diff
pub fn overlapping_lifetimes(flag: bool) -> Fields {
    // This checks the liveness-matrix overlap test for an address-observed
    // move-only local: `src` and `dst` only overlap on one branch, but that is
    // enough to reject merging them for the whole function.
    // CHECK-LABEL: fn overlapping_lifetimes(
    // CHECK: debug flag => _1;
    // CHECK: debug src => [[src:_[1-9][0-9]*]];
    // CHECK: debug dst => _0;
    // CHECK: &raw const [[src]];
    // CHECK: observe
    // CHECK: switchInt(move _1)
    // CHECK: _0 = make_fields(const 1_u8)
    // CHECK: _0 = move [[src]];
    // CHECK: &raw const _0;
    // CHECK: observe
    let src = make_fields(0);
    let mut dst;
    observe(&raw const src);
    if flag {
        dst = make_fields(1);
        let _ = src.tag;
    } else {
        dst = src;
    }
    observe(&raw const dst);
    dst
}

// EMIT_MIR exclusions.dse_guard.MoveElimination.diff
pub fn dse_guard() {
    // This guards the RFC soundness hazard: DSE must not remove the first write
    // to `b`, because that write keeps `b`'s address-observed lifetime
    // overlapping with `a` and prevents the later move from being eliminated.
    // CHECK-LABEL: fn dse_guard(
    // CHECK: StorageLive([[b:_.*]]);
    // CHECK: [[b]] = make_fields(const 0_u8)
    // CHECK: StorageLive([[a:_.*]]);
    // CHECK: [[a]] = make_fields(const 1_u8)
    // CHECK: observe(move
    // CHECK: [[b]] = move [[a]]
    // CHECK: observe(move
    let mut a;
    let mut b;

    b = make_fields(0);

    a = make_fields(1);
    observe(&raw const a);
    b = a;
    observe(&raw const b);
}

// EMIT_MIR exclusions.rust_call_tuple_not_projected.MoveElimination.diff
pub fn rust_call_tuple_not_projected<F: FnOnce([u8; 8], [u8; 8])>(f: F) {
    // This checks that locals are not remapped into the tuple argument passed
    // to a rust-call ABI function. If the tuple itself is never borrowed, alias
    // analysis can trivially see that accesses to one argument don't affect the
    // others. Merging the arguments into tuple fields from the start can hide
    // that independence.
    // CHECK-LABEL: fn rust_call_tuple_not_projected(
    // CHECK: debug a => [[a:_.*]];
    // CHECK: debug b => [[b:_.*]];
    // CHECK: [[tuple:_.*]] = (move [[a]], move [[b]]);
    // CHECK: <F as FnOnce<([u8; 8], [u8; 8])>>::call_once(move _1, move [[tuple]])
    let a = [1; 8];
    let b = [2; 8];
    f(a, b);
}
