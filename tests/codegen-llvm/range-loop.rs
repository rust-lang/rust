//@ ignore-std-debug-assertions
//@ compile-flags: -Copt-level=3 -C no-prepopulate-passes

#![crate_type = "lib"]

// Ensure that MIR optimizations have cleaned things up enough that the IR we
// emit is good even without running the LLVM optimizations.

// CHECK-NOT: define

// CHECK-LABEL: define{{.+}}void @call_for_zero_to_n
#[no_mangle]
pub fn call_for_zero_to_n(n: u32, f: fn(u32)) {
    // CHECK: start:
    // CHECK-NOT: alloca
    // CHECK: %[[IND:.+]] = alloca [4 x i8]
    // CHECK-NEXT: %[[ALWAYS_SOME_OPTION:.+]] = alloca
    // CHECK-NOT: alloca
    // CHECK: store i32 0, ptr %[[IND]],
    // CHECK: br label %[[HEAD:.+]]

    // CHECK: [[HEAD]]:
    // CHECK: %[[T1:.+]] = load i32, ptr %[[IND]],
    // CHECK: %[[NOT_DONE:.+]] = icmp ult i32 %[[T1]], %n
    // CHECK: br i1 %[[NOT_DONE]], label %[[BODY:.+]], label %[[BREAK:.+]]

    // CHECK: [[BREAK]]:
    // CHECK: ret void

    // CHECK: [[BODY]]:
    // CHECK: %[[T2:.+]] = load i32, ptr %[[IND]],
    // CHECK: %[[T3:.+]] = add nuw i32 %[[T2]], 1
    // CHECK: store i32 %[[T3]], ptr %[[IND]],

    // CHECK: store i32 %[[T2]]
    // CHECK: %[[T4:.+]] = load i32
    // CHECK: call void %f(i32{{.+}}%[[T4]])

    for i in 0..n {
        f(i);
    }
}

// CHECK-NOT: define
