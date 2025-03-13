//@ compile-flags: -Copt-level=3 -C no-prepopulate-passes

#![crate_type = "lib"]

// CHECK-LABEL: @simple_is_one_block
#[no_mangle]
pub unsafe fn simple_is_one_block(x: i32) -> i32 {
    // CHECK: start:
    // CHECK-NEXT: ret i32 %x

    // CHECK-NOT: return

    x
}

// CHECK-LABEL: @branch_has_shared_block
#[no_mangle]
pub unsafe fn branch_has_shared_block(b: bool) -> i32 {
    // CHECK: start:
    // CHECK-NEXT: %[[A:.+]] = alloca [4 x i8]
    // CHECK-NEXT: br i1 %b

    // CHECK: store i32 {{42|2015}}, ptr %[[A]]
    // CHECK-NEXT: br label %return

    // CHECK: store i32 {{42|2015}}, ptr %[[A]]
    // CHECK-NEXT: br label %return

    // CHECK: return:
    // CHECK-NEXT: %[[R:.+]] = load i32, ptr %[[A]]
    // CHECK-NEXT: ret i32 %[[R]]

    if b { 42 } else { 2015 }
}
