//@ compile-flags: -Cpanic=abort
//@ test-mir-pass: GVN

fn opaque<T>(x: T) {}

// EMIT_MIR borrowed_variable.borrow.GVN.diff
fn borrow() {
    // CHECK-LABEL: fn borrow(
    // CHECK: debug x => [[x:_.*]];
    // CHECK: debug r => [[r:_.*]];
    // CHECK: [[x]] = const 42_i32;
    // CHECK: opaque::<i32>(const 42_i32)
    // CHECK: [[r]] = &[[x]];
    // CHECK: opaque::<i32>(const 42_i32)
    let mut x = 42;
    opaque(x);
    let r = &x;
    opaque(x);
}

// EMIT_MIR borrowed_variable.nonfreeze.GVN.diff
fn nonfreeze<T: Copy>(mut x: T) {
    // CHECK-LABEL: fn nonfreeze(
    // CHECK: debug x => [[x:_.*]];
    // CHECK: debug y => [[y:_.*]];
    // CHECK: debug r => [[r:_.*]];
    // CHECK: [[y]] = copy [[x]];
    // The following line prefers reusing `y` which is SSA.
    // CHECK: opaque::<T>(copy [[y]])
    // CHECK: [[r]] = &[[x]];
    // CHECK: [[tmpx:_.*]] = copy [[x]];
    // CHECK: opaque::<T>(move [[tmpx]])
    // CHECK: opaque::<T>(copy [[y]])
    let mut y = x;
    opaque(y);
    let r = &x;
    opaque(x);
    opaque(y);
}

// EMIT_MIR borrowed_variable.mutborrow.GVN.diff
fn mutborrow() {
    // CHECK-LABEL: fn mutborrow(
    // CHECK: debug x => [[x:_.*]];
    // CHECK: debug r => [[r:_.*]];
    // CHECK: [[x]] = const 42_i32;
    // CHECK: opaque::<i32>(const 42_i32)
    // CHECK: [[r]] = &mut [[x]];
    // CHECK: [[tmp:_.*]] = copy [[x]];
    // CHECK: opaque::<i32>(move [[tmp]])
    let mut x = 42;
    opaque(x);
    let r = &mut x;
    opaque(x);
}

// EMIT_MIR borrowed_variable.constptr.GVN.diff
fn constptr() {
    // CHECK-LABEL: fn constptr(
    // CHECK: debug x => [[x:_.*]];
    // CHECK: debug r => [[r:_.*]];
    // CHECK: [[x]] = const 42_i32;
    // CHECK: opaque::<i32>(const 42_i32)
    // CHECK: [[r]] = &raw const [[x]];
    // CHECK: [[tmp:_.*]] = copy [[x]];
    // CHECK: opaque::<i32>(move [[tmp]])
    let mut x = 42;
    opaque(x);
    let r = &raw const x;
    opaque(x);
}

// EMIT_MIR borrowed_variable.mutptr.GVN.diff
fn mutptr() {
    // CHECK-LABEL: fn mutptr(
    // CHECK: debug x => [[x:_.*]];
    // CHECK: debug r => [[r:_.*]];
    // CHECK: [[x]] = const 42_i32;
    // CHECK: opaque::<i32>(const 42_i32)
    // CHECK: [[r]] = &raw mut [[x]];
    // CHECK: [[tmp:_.*]] = copy [[x]];
    // CHECK: opaque::<i32>(move [[tmp]])
    let mut x = 42;
    opaque(x);
    let r = &raw mut x;
    opaque(x);
}
