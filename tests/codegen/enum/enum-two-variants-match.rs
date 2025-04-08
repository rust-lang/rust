//@ compile-flags: -Copt-level=3 -C no-prepopulate-passes
//@ only-64bit (because these discriminants are isize)

#![crate_type = "lib"]

// This directly tests what we emit for these matches, rather than what happens
// after optimization, so it doesn't need to worry about extra flags on the
// instructions and is less susceptible to being broken on LLVM updates.

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

// CHECK-LABEL: @option_bool_match(
#[no_mangle]
pub fn option_bool_match(x: Option<bool>) -> char {
    // CHECK: %[[RAW:.+]] = load i8, ptr %x
    // CHECK: %[[IS_NONE:.+]] = icmp eq i8 %[[RAW]], 2
    // CHECK: %[[OPT_DISCR:.+]] = select i1 %[[IS_NONE]], i64 0, i64 1
    // CHECK: %[[OPT_DISCR_T:.+]] = trunc nuw i64 %[[OPT_DISCR]] to i1
    // CHECK: br i1 %[[OPT_DISCR_T]], label %[[BB_SOME:.+]], label %[[BB_NONE:.+]]

    // CHECK: [[BB_SOME]]:
    // CHECK: %[[FIELD:.+]] = load i8, ptr %x
    // CHECK: %[[FIELD_T:.+]] = trunc nuw i8 %[[FIELD]] to i1
    // CHECK: br i1 %[[FIELD_T]]
    match x {
        None => 'n',
        Some(false) => 'f',
        Some(true) => 't',
    }
}

use std::cmp::Ordering::{self, *};
// CHECK-LABEL: @option_ordering_match(
#[no_mangle]
pub fn option_ordering_match(x: Option<Ordering>) -> char {
    // CHECK: %[[RAW:.+]] = load i8, ptr %x
    // CHECK: %[[IS_NONE:.+]] = icmp eq i8 %[[RAW]], 2
    // CHECK: %[[OPT_DISCR:.+]] = select i1 %[[IS_NONE]], i64 0, i64 1
    // CHECK: %[[OPT_DISCR_T:.+]] = trunc nuw i64 %[[OPT_DISCR]] to i1
    // CHECK: br i1 %[[OPT_DISCR_T]], label %[[BB_SOME:.+]], label %[[BB_NONE:.+]]

    // CHECK: [[BB_SOME]]:
    // CHECK: %[[FIELD:.+]] = load i8, ptr %x
    // CHECK: switch i8 %[[FIELD]], label %[[UNREACHABLE:.+]] [
    // CHECK-NEXT: i8 -1, label
    // CHECK-NEXT: i8 0, label
    // CHECK-NEXT: i8 1, label
    // CHECK-NEXT: ]

    // CHECK: [[UNREACHABLE]]:
    // CHECK-NEXT: unreachable
    match x {
        None => '?',
        Some(Less) => '<',
        Some(Equal) => '=',
        Some(Greater) => '>',
    }
}

// CHECK-LABEL: @option_nonzero_match(
#[no_mangle]
pub fn option_nonzero_match(x: Option<std::num::NonZero<u16>>) -> u16 {
    // CHECK: %[[OUT:.+]] = alloca [2 x i8]

    // CHECK: %[[IS_NONE:.+]] = icmp eq i16 %x, 0
    // CHECK: %[[OPT_DISCR:.+]] = select i1 %[[IS_NONE]], i64 0, i64 1
    // CHECK: %[[OPT_DISCR_T:.+]] = trunc nuw i64 %[[OPT_DISCR]] to i1
    // CHECK: br i1 %[[OPT_DISCR_T]], label %[[BB_SOME:.+]], label %[[BB_NONE:.+]]

    // CHECK: [[BB_SOME]]:
    // CHECK: store i16 987, ptr %[[OUT]]

    // CHECK: [[BB_NONE]]:
    // CHECK: store i16 123, ptr %[[OUT]]

    // CHECK: %[[RET:.+]] = load i16, ptr %[[OUT]]
    // CHECK: ret i16 %[[RET]]

    match x {
        None => 123,
        Some(_) => 987,
    }
}
