// GVN may create indirect constants with higher alignment than their type requires. Verify that we
// do not ICE during codegen, and that the LLVM constant has the higher alignment.
//
//@ compile-flags: -Zmir-opt-level=0 -Zmir-enable-passes=+GVN
//@ compile-flags: -Cno-prepopulate-passes
//@ only-64bit

struct S(i32);

struct SmallStruct(f32, Option<S>, &'static [f32]);

// CHECK: @0 = private unnamed_addr constant
// CHECK-SAME: , align 8

fn main() {
    // CHECK-LABEL: @_ZN20overaligned_constant4main
    // CHECK: [[full:%_.*]] = alloca %SmallStruct, align 8
    // CHECK: call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[full]], ptr align 8 @0, i64 32, i1 false)
    // CHECK: %b.0 = load i32, ptr @0, align 4,
    // CHECK: %b.1 = load i32, ptr getelementptr inbounds ({{.*}}), align 4
    let mut s = S(1);

    s.0 = 3;

    // SMALL_VAL corresponds to a MIR allocation with alignment 8.
    const SMALL_VAL: SmallStruct = SmallStruct(4., Some(S(1)), &[]);

    // In pre-codegen MIR:
    // `a` is a scalar 4.
    // `b` is an indirect constant at `SMALL_VAL`'s alloc with 0 offset.
    // `c` is the empty slice.
    //
    // As a consequence, during codegen, we create a LLVM allocation for `SMALL_VAL`, with
    // alignment 8, but only use the `Option<S>` field, at offset 0 with alignment 4.
    let SmallStruct(a, b, c) = SMALL_VAL;
}
