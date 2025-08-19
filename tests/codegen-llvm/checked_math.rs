//@ compile-flags: -Copt-level=3 -Z merge-functions=disabled

#![crate_type = "lib"]
#![feature(unchecked_shifts)]

// Because the result of something like `u32::checked_sub` can only be used if it
// didn't overflow, make sure that LLVM actually knows that in optimized builds.
// Thanks to poison semantics, this doesn't even need branches.

// CHECK-LABEL: @checked_sub_unsigned
// CHECK-SAME: (i16{{.*}} %a, i16{{.*}} %b)
#[no_mangle]
pub fn checked_sub_unsigned(a: u16, b: u16) -> Option<u16> {
    // CHECK-DAG: %[[IS_SOME:.+]] = icmp uge i16 %a, %b
    // CHECK-DAG: %[[DIFF_P:.+]] = sub nuw i16 %a, %b
    // CHECK-DAG: %[[DISCR:.+]] = zext i1 %[[IS_SOME]] to i16
    // CHECK-DAG: %[[DIFF_U:.+]] = select i1 %[[IS_SOME]], i16 %[[DIFF_P]], i16 undef

    // CHECK: %[[R0:.+]] = insertvalue { i16, i16 } poison, i16 %[[DISCR]], 0
    // CHECK: %[[R1:.+]] = insertvalue { i16, i16 } %[[R0]], i16 %[[DIFF_U]], 1
    // CHECK: ret { i16, i16 } %[[R1]]
    a.checked_sub(b)
}

// Note that `shl` and `shr` in LLVM are already unchecked. So rather than
// looking for no-wrap flags, we just need there to not be any masking.

// CHECK-LABEL: @checked_shl_unsigned
// CHECK-SAME: (i32{{.*}} %a, i32{{.*}} %b)
#[no_mangle]
pub fn checked_shl_unsigned(a: u32, b: u32) -> Option<u32> {
    // CHECK-DAG: %[[IS_SOME:.+]] = icmp ult i32 %b, 32
    // CHECK-DAG: %[[SHIFTED_P:.+]] = shl i32 %a, %b
    // CHECK-DAG: %[[DISCR:.+]] = zext i1 %[[IS_SOME]] to i32
    // CHECK-DAG: %[[SHIFTED_U:.+]] = select i1 %[[IS_SOME]], i32 %[[SHIFTED_P]], i32 undef

    // CHECK: %[[R0:.+]] = insertvalue { i32, i32 } poison, i32 %[[DISCR]], 0
    // CHECK: %[[R1:.+]] = insertvalue { i32, i32 } %[[R0]], i32 %[[SHIFTED_U]], 1
    // CHECK: ret { i32, i32 } %[[R1]]
    a.checked_shl(b)
}

// CHECK-LABEL: @checked_shr_unsigned
// CHECK-SAME: (i32{{.*}} %a, i32{{.*}} %b)
#[no_mangle]
pub fn checked_shr_unsigned(a: u32, b: u32) -> Option<u32> {
    // CHECK-DAG: %[[IS_SOME:.+]] = icmp ult i32 %b, 32
    // CHECK-DAG: %[[SHIFTED_P:.+]] = lshr i32 %a, %b
    // CHECK-DAG: %[[DISCR:.+]] = zext i1 %[[IS_SOME]] to i32
    // CHECK-DAG: %[[SHIFTED_U:.+]] = select i1 %[[IS_SOME]], i32 %[[SHIFTED_P]], i32 undef

    // CHECK: %[[R0:.+]] = insertvalue { i32, i32 } poison, i32 %[[DISCR]], 0
    // CHECK: %[[R1:.+]] = insertvalue { i32, i32 } %[[R0]], i32 %[[SHIFTED_U]], 1
    // CHECK: ret { i32, i32 } %[[R1]]
    a.checked_shr(b)
}

// CHECK-LABEL: @checked_shl_signed
// CHECK-SAME: (i32{{.*}} %a, i32{{.*}} %b)
#[no_mangle]
pub fn checked_shl_signed(a: i32, b: u32) -> Option<i32> {
    // CHECK-DAG: %[[IS_SOME:.+]] = icmp ult i32 %b, 32
    // CHECK-DAG: %[[SHIFTED_P:.+]] = shl i32 %a, %b
    // CHECK-DAG: %[[DISCR:.+]] = zext i1 %[[IS_SOME]] to i32
    // CHECK-DAG: %[[SHIFTED_U:.+]] = select i1 %[[IS_SOME]], i32 %[[SHIFTED_P]], i32 undef

    // CHECK: %[[R0:.+]] = insertvalue { i32, i32 } poison, i32 %[[DISCR]], 0
    // CHECK: %[[R1:.+]] = insertvalue { i32, i32 } %[[R0]], i32 %[[SHIFTED_U]], 1
    // CHECK: ret { i32, i32 } %[[R1]]
    a.checked_shl(b)
}

// CHECK-LABEL: @checked_shr_signed
// CHECK-SAME: (i32{{.*}} %a, i32{{.*}} %b)
#[no_mangle]
pub fn checked_shr_signed(a: i32, b: u32) -> Option<i32> {
    // CHECK-DAG: %[[IS_SOME:.+]] = icmp ult i32 %b, 32
    // CHECK-DAG: %[[SHIFTED_P:.+]] = ashr i32 %a, %b
    // CHECK-DAG: %[[DISCR:.+]] = zext i1 %[[IS_SOME]] to i32
    // CHECK-DAG: %[[SHIFTED_U:.+]] = select i1 %[[IS_SOME]], i32 %[[SHIFTED_P]], i32 undef

    // CHECK: %[[R0:.+]] = insertvalue { i32, i32 } poison, i32 %[[DISCR]], 0
    // CHECK: %[[R1:.+]] = insertvalue { i32, i32 } %[[R0]], i32 %[[SHIFTED_U]], 1
    // CHECK: ret { i32, i32 } %[[R1]]
    a.checked_shr(b)
}

// CHECK-LABEL: @checked_add_one_unwrap_unsigned
// CHECK-SAME: (i32{{.*}} %x)
#[no_mangle]
pub fn checked_add_one_unwrap_unsigned(x: u32) -> u32 {
    // CHECK: %[[IS_MAX:.+]] = icmp eq i32 %x, -1
    // CHECK: br i1 %[[IS_MAX]], label %[[NONE_BB:.+]], label %[[SOME_BB:.+]],
    // CHECK: [[SOME_BB]]:
    // CHECK: %[[R:.+]] = add nuw i32 %x, 1
    // CHECK: ret i32 %[[R]]
    // CHECK: [[NONE_BB]]:
    // CHECK: call {{.+}}unwrap_failed
    x.checked_add(1).unwrap()
}
