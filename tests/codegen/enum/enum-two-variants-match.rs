//@ compile-flags: -Copt-level=3 -C no-prepopulate-passes
//@ min-llvm-version: 19 (for trunc nuw)
//@ only-x86_64 (because these discriminants are isize)

#![crate_type = "lib"]

// CHECK-LABEL: @option_match
#[no_mangle]
pub fn option_match(x: Option<i32>) -> u16 {
    // CHECK: %x = alloca [8 x i8]
    // CHECK: store i32 %0, ptr %x
    // CHECK: %[[TAG:.+]] = load i32, ptr %x
    // CHECK-SAME: !range ![[ZERO_ONE_32:[0-9]+]]
    // CHECK: %[[DISCR:.+]] = zext i32 %[[TAG]] to i64
    // CHECK: %[[COND:.+]] = trunc nuw i64 %[[DISCR]] to i1
    // CHECK: br i1 %[[COND]], label %[[TRUE:[a-z0-9]+]], label %[[FALSE:[a-z0-9]+]]

    // CHECK: [[TRUE]]:
    // CHECK: store i16 13

    // CHECK: [[FALSE]]:
    // CHECK: store i16 42
    match x {
        Some(_) => 13,
        None => 42,
    }
}

// CHECK-LABEL: @result_match
#[no_mangle]
pub fn result_match(x: Result<u64, i64>) -> u16 {
    // CHECK: %x = alloca [16 x i8]
    // CHECK: store i64 %0, ptr %x
    // CHECK: %[[DISCR:.+]] = load i64, ptr %x
    // CHECK-SAME: !range ![[ZERO_ONE_64:[0-9]+]]
    // CHECK: %[[COND:.+]] = trunc nuw i64 %[[DISCR]] to i1
    // CHECK: br i1 %[[COND]], label %[[TRUE:[a-z0-9]+]], label %[[FALSE:[a-z0-9]+]]

    // CHECK: [[TRUE]]:
    // CHECK: store i16 13

    // CHECK: [[FALSE]]:
    // CHECK: store i16 42
    match x {
        Err(_) => 13,
        Ok(_) => 42,
    }
}

// CHECK: ![[ZERO_ONE_32]] = !{i32 0, i32 2}
// CHECK: ![[ZERO_ONE_64]] = !{i64 0, i64 2}
