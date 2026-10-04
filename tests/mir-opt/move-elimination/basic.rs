//@ test-mir-pass: MoveElimination
//@ compile-flags: -Cpanic=abort -Zmir-enable-passes=+TailCopyToMove

struct Pair {
    a: [u8; 8],
    b: [u8; 8],
}

fn init(_: &mut [u8; 8]) {}

// EMIT_MIR basic.nrvo_unborrowed.MoveElimination.diff
pub fn nrvo_unborrowed() -> [u8; 8] {
    // This checks the simplest NRVO-style case: the local should be merged with
    // the return place even though it has `Copy` type.
    // CHECK-LABEL: fn nrvo_unborrowed(
    // CHECK: debug buf => _0;
    // CHECK: _0 = [const 1_u8; 8]
    let buf = [1; 8];
    buf
}

// EMIT_MIR basic.nrvo_borrowed.MoveElimination.diff
pub fn nrvo_borrowed() -> [u8; 8] {
    // This checks that taking a temporary mutable borrow does not prevent
    // merging a `Copy` local once the borrow has ended.
    // CHECK-LABEL: fn nrvo_borrowed(
    // CHECK: debug buf => _0;
    // CHECK: _0 = [const 1_u8; 8]
    // CHECK: init(move {{_.*}})
    // CHECK-NOT: _0 = move
    let mut buf = [1; 8];
    init(&mut buf);
    buf
}

// EMIT_MIR basic.struct_aggregate.MoveElimination.diff
pub fn struct_aggregate() -> Pair {
    // This checks aggregate field remapping: the field locals can live directly
    // in the return place's fields.
    // CHECK-LABEL: fn struct_aggregate(
    // CHECK: debug a => (_0.0: [u8; 8]);
    // CHECK: debug b => (_0.1: [u8; 8]);
    // CHECK: (_0.0: [u8; 8]) = [const 1_u8; 8];
    // CHECK: (_0.1: [u8; 8]) = [const 2_u8; 8];
    let a = [1; 8];
    let b = [2; 8];
    Pair { a, b }
}

// EMIT_MIR basic.enum_aggregate.MoveElimination.diff
pub fn enum_aggregate() -> Result<([u8; 8], [u8; 8]), ()> {
    // This checks aggregate field remapping for enums: the payload fields can
    // be written directly and then the discriminant is set for the variant.
    // CHECK-LABEL: fn enum_aggregate(
    // CHECK: debug a => (((_0 as variant#0).0: ([u8; 8], [u8; 8])).0: [u8; 8]);
    // CHECK: debug b => (((_0 as variant#0).0: ([u8; 8], [u8; 8])).1: [u8; 8]);
    // CHECK: (((_0 as variant#0).0: ([u8; 8], [u8; 8])).0: [u8; 8]) = [const 1_u8; 8];
    // CHECK: (((_0 as variant#0).0: ([u8; 8], [u8; 8])).1: [u8; 8]) = [const 2_u8; 8];
    // CHECK: discriminant(_0) = 0;
    let a = [1; 8];
    let b = [2; 8];
    Result::Ok((a, b))
}

// EMIT_MIR basic.array_aggregate.MoveElimination.diff
pub fn array_aggregate() -> [[u8; 8]; 3] {
    // This checks aggregate remapping for arrays, which uses ConstantIndex
    // projections rather than field projections.
    // CHECK-LABEL: fn array_aggregate(
    // CHECK: debug a => _0[0 of 1];
    // CHECK: debug b => _0[1 of 2];
    // CHECK: debug c => _0[2 of 3];
    // CHECK: _0[0 of 1] = [const 1_u8; 8];
    // CHECK: _0[1 of 2] = [const 2_u8; 8];
    // CHECK: _0[2 of 3] = [const 3_u8; 8];
    let a = [1; 8];
    let b = [2; 8];
    let c = [3; 8];
    [a, b, c]
}
