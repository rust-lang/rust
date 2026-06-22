//@ test-mir-pass: MoveElimination
//@ compile-flags: -Cpanic=abort -Zmir-enable-passes=+DeadStoreElimination-initial

pub struct Fields {
    data: [u8; 8],
    tag: u8,
}

unsafe extern "C" {
    safe fn observe(_: *const Fields);
    safe fn make_fields(_: u8) -> Fields;
}

// EMIT_MIR dse.dse_guard.MoveElimination.diff
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
