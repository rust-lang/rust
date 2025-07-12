//@ test-mir-pass: DeadStoreElimination-initial

pub struct Foo {
    a: i32,
    b: i64,
    c: i32,
}

// EMIT_MIR ref.tuple.DeadStoreElimination-initial.diff
pub fn tuple(v: (i32, &Foo)) -> i32 {
    // CHECK-LABEL: fn tuple
    // CHECK: debug _dead => [[dead:_[0-9]+]];
    // CHECK: bb0:
    // FIXME: Preserve `tmp` for debuginfo, but we can merge it into the debuginfo.
    // CHECK-NEXT: [[tmp:_[0-9]+]] = deref_copy (_1.1: &Foo);
    // CHECK-NEXT: DBG: AssignRef([[dead]], ((*[[tmp]]).2: i32))
    let _dead = &v.1.c;
    v.1.a
}
