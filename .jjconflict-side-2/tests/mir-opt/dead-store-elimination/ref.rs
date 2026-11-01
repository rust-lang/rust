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

// EMIT_MIR ref.dead_first.DeadStoreElimination-initial.diff
pub fn dead_first(v: &Foo) -> &i32 {
    // CHECK-LABEL: fn dead_first
    // CHECK: debug a => [[var_a:_[0-9]+]];
    // CHECK: bb0:
    // CHECK: DBG: [[var_a]] = &((*_1).2: i32)
    // CHECK: [[tmp_4:_[0-9]+]] = &((*_1).0: i32)
    // CHECK: [[tmp_3:_[0-9]+]] = &(*[[tmp_4]])
    // CHECK: [[var_a]] = move [[tmp_3]]
    let mut a = &v.c;
    a = &v.a;
    a
}
