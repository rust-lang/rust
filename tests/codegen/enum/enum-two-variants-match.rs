//@ compile-flags: -Copt-level=3 -C no-prepopulate-passes
//@ only-x86_64 (because these discriminants are isize)

#![crate_type = "lib"]

// CHECK-LABEL: @option_match
#[no_mangle]
pub fn option_match(x: Option<i32>) -> u16 {
    // CHECK-NOT: %x = alloca
    // CHECK: %[[OUT:.+]] = alloca [2 x i8]
    // CHECK-NOT: %x = alloca

    // CHECK: %[[DISCR:.+]] = zext i32 %x.0 to i64
    // CHECK: %[[COND:.+]] = trunc nuw i64 %[[DISCR]] to i1
    // CHECK: br i1 %[[COND]], label %[[TRUE:[a-z0-9]+]], label %[[FALSE:[a-z0-9]+]]

    // CHECK: [[TRUE]]:
    // CHECK: store i16 13, ptr %[[OUT]]

    // CHECK: [[FALSE]]:
    // CHECK: store i16 42, ptr %[[OUT]]

    // CHECK: %[[RET:.+]] = load i16, ptr %[[OUT]]
    // CHECK: ret i16 %[[RET]]
    match x {
        Some(_) => 13,
        None => 42,
    }
}

// CHECK-LABEL: @result_match
#[no_mangle]
pub fn result_match(x: Result<u64, i64>) -> u16 {
    // CHECK-NOT: %x = alloca
    // CHECK: %[[OUT:.+]] = alloca [2 x i8]
    // CHECK-NOT: %x = alloca

    // CHECK: %[[COND:.+]] = trunc nuw i64 %x.0 to i1
    // CHECK: br i1 %[[COND]], label %[[TRUE:[a-z0-9]+]], label %[[FALSE:[a-z0-9]+]]

    // CHECK: [[TRUE]]:
    // CHECK: store i16 13, ptr %[[OUT]]

    // CHECK: [[FALSE]]:
    // CHECK: store i16 42, ptr %[[OUT]]

    // CHECK: %[[RET:.+]] = load i16, ptr %[[OUT]]
    // CHECK: ret i16 %[[RET]]
    match x {
        Err(_) => 13,
        Ok(_) => 42,
    }
}
