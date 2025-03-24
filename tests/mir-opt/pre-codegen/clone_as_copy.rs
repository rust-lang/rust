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
    // CHECK-NOT: = AllCopy { {{.*}} };
    // CHECK-NOT: = NestCopy { {{.*}} };
    // CHECK: _0 = copy (*_1);
    // CHECK: return;
    v.clone()
}

// FIXME: We can merge into exactly one assignment statement.
// EMIT_MIR clone_as_copy.enum_clone_as_copy.PreCodegen.after.mir
fn enum_clone_as_copy(v: &Enum1) -> Enum1 {
    // CHECK-LABEL: fn enum_clone_as_copy(
    // CHECK-NOT: = Enum1::
    // CHECK: _0 = copy (*_1);
    // CHECK: _0 = copy (*_1);
    v.clone()
}
