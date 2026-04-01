//@ compile-flags: -Cdebuginfo=full

// Check if we have transformed the nested clone to the copy in the complete pipeline.

#[derive(Clone)]
struct AllCopy {
    a: i32,
    b: u64,
    c: [i8; 3],
}

#[derive(Clone)]
struct NestCopy {
    a: i32,
    b: AllCopy,
    c: [i8; 3],
}

#[derive(Clone)]
enum Enum1 {
    A(AllCopy),
    B(NestCopy),
}

// EMIT_MIR clone_as_copy.clone_as_copy.PreCodegen.after.mir
fn clone_as_copy(v: &NestCopy) -> NestCopy {
    // CHECK-LABEL: fn clone_as_copy(
    // CHECK: let [[DEAD_VAR:_.*]]: &AllCopy;
    // CHECK: bb0: {
    // CHECK-NEXT: DBG: [[DEAD_VAR]] = &((*_1).1: AllCopy)
    // CHECK-NEXT: _0 = copy (*_1);
    // CHECK-NEXT: return;
    v.clone()
}

// EMIT_MIR clone_as_copy.enum_clone_as_copy.PreCodegen.after.mir
fn enum_clone_as_copy(v: &Enum1) -> Enum1 {
    // CHECK-LABEL: fn enum_clone_as_copy(
    // CHECK: bb0: {
    // CHECK-NEXT: _0 = copy (*_1);
    // CHECK-NEXT: return;
    v.clone()
}
