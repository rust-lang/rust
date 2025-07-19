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
    // CHECK: DBG: [[dead]] = &((*_3).2: i32)
    let _dead = &v.1.c;
    v.1.a
}
