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

// EMIT_MIR clone_as_copy.{impl#0}-clone.PreCodegen.after.mir
// CHECK-LABEL: fn <impl at {{.*}}>::clone(_1: &AllCopy) -> AllCopy {
// CHECK: bb0: {
// CHECK-NEXT: _0 = copy (*_1);
// CHECK-NEXT: return;

// EMIT_MIR clone_as_copy.{impl#1}-clone.PreCodegen.after.mir
// CHECK-LABEL: fn <impl at {{.*}}>::clone(_1: &NestCopy) -> NestCopy {
// CHECK: scope 1 (inlined <AllCopy as Clone>::clone) {
// CHECK-NEXT: debug self => [[inlined_AllCopy_self:_[0-9]+]];
// CHECK: bb0: {
// CHECK-NEXT: DBG: [[inlined_AllCopy_self]] = &((*_1).1: AllCopy)
// CHECK-NEXT: _0 = copy (*_1);
// CHECK-NEXT: return;

// EMIT_MIR clone_as_copy.{impl#2}-clone.PreCodegen.after.mir
// CHECK-LABEL: fn <impl at {{.*}}>::clone(_1: &Enum1) -> Enum1 {
// CHECK: bb0: {
// CHECK-NEXT: _0 = copy (*_1);
// CHECK-NEXT: return;
