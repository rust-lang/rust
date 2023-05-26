// compile-flags: --crate-type=lib -Copt-level=0 -Zmir-opt-level=0 -C debuginfo=2

// CHECK: @0 = private unnamed_addr constant <{ [8 x i8] }> <{ [8 x i8] c"\01\00\00\00\02\00\00\00" }>, align 4

#![feature(inline_const)]

pub fn foo() -> (i32, i32) {
    // CHECK: %0 = load i32, ptr @0, align 4
    const { (1, 2) }
}
