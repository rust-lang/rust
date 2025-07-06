//@ revisions: OPT DBG
//@ compile-flags: -Cno-prepopulate-passes -Cdebuginfo=0
//@[OPT] compile-flags: -Copt-level=1
//@[DBG] compile-flags: -Copt-level=0
//@ min-llvm-version: 19
//@ only-64bit

#![crate_type = "lib"]

// This tests various cases around consuming enums as SSA values in what we emit.
// Importantly, it checks things like correct `i1` handling for `bool`
// and for mixed integer/pointer payloads.

use std::cmp::Ordering;
use std::mem::MaybeUninit;
use std::num::NonZero;
use std::ptr::NonNull;

// This doesn't actually end up in an SSA value because `extract_field`
// doesn't know how to do the equivalent of `!noundef` without a load.
#[no_mangle]
fn use_option_u32(x: Option<u32>) -> u32 {
    // CHECK-LABEL: @use_option_u32
    // OPT-SAME: (i32 noundef range(i32 0, 2) %0, i32 %1)

    // CHECK-NOT: alloca
    // CHECK: %x = alloca [8 x i8]
    // CHECK-NOT: alloca
    // CHECK: %[[X0:.+]] = load i32, ptr %x
    // CHECK: %[[DISCR:.+]] = zext i32 %[[X0]] to i64
    // CHECK: %[[IS_SOME:.+]] = trunc nuw i64 %[[DISCR]] to i1
    // OPT: %[[LIKELY:.+]] = call i1 @llvm.expect.i1(i1 %[[IS_SOME]], i1 true)
    // OPT: br i1 %[[LIKELY]], label %[[BLOCK:.+]],
    // DBG: br i1 %[[IS_SOME]], label %[[BLOCK:.+]],

    // CHECK: [[BLOCK]]:
    // CHECK: %[[X1P:.+]] = getelementptr inbounds i8, ptr %x, i64 4
    // CHECK: %[[X1:.+]] = load i32, ptr %[[X1P]]
    // OPT-SAME: !noundef
    // CHECK: ret i32 %[[X1]]

    if let Some(val) = x { val } else { unreachable!() }
}

#[no_mangle]
fn use_option_mu_i32(x: Option<MaybeUninit<i32>>) -> MaybeUninit<i32> {
    // CHECK-LABEL: @use_option_mu_i32
    // OPT-SAME: (i32 noundef range(i32 0, 2) %x.0, i32 %x.1)

    // CHECK-NOT: alloca
    // CHECK: %[[DISCR:.+]] = zext i32 %x.0 to i64
    // CHECK: %[[IS_SOME:.+]] = trunc nuw i64 %[[DISCR]] to i1
    // OPT: %[[LIKELY:.+]] = call i1 @llvm.expect.i1(i1 %[[IS_SOME]], i1 true)
    // OPT: br i1 %[[LIKELY]], label %[[BLOCK:.+]],
    // DBG: br i1 %[[IS_SOME]], label %[[BLOCK:.+]],

    // CHECK: [[BLOCK]]:
    // CHECK: ret i32 %x.1

    if let Some(val) = x { val } else { unreachable!() }
}

#[no_mangle]
fn use_option_bool(x: Option<bool>) -> bool {
    // CHECK-LABEL: @use_option_bool
    // OPT-SAME: (i8 noundef range(i8 0, 3) %x)

    // CHECK-NOT: alloca
    // CHECK: %[[IS_NONE:.+]] = icmp eq i8 %x, 2
    // CHECK: %[[DISCR:.+]] = select i1 %[[IS_NONE]], i64 0, i64 1
    // CHECK: %[[IS_SOME:.+]] = trunc nuw i64 %[[DISCR]] to i1
    // OPT: %[[LIKELY:.+]] = call i1 @llvm.expect.i1(i1 %[[IS_SOME]], i1 true)
    // OPT: br i1 %[[LIKELY]], label %[[BLOCK:.+]],
    // DBG: br i1 %[[IS_SOME]], label %[[BLOCK:.+]],

    // CHECK: [[BLOCK]]:
    // CHECK: %val = trunc nuw i8 %x to i1
    // CHECK: ret i1 %val

    if let Some(val) = x { val } else { unreachable!() }
}

#[no_mangle]
fn use_option_ordering(x: Option<Ordering>) -> Ordering {
    // CHECK-LABEL: @use_option_ordering
    // OPT-SAME: (i8 noundef range(i8 -1, 3) %x)

    // CHECK: %[[IS_NONE:.+]] = icmp eq i8 %x, 2
    // CHECK: %[[DISCR:.+]] = select i1 %[[IS_NONE]], i64 0, i64 1
    // CHECK: %[[IS_SOME:.+]] = trunc nuw i64 %[[DISCR]] to i1
    // OPT: %[[LIKELY:.+]] = call i1 @llvm.expect.i1(i1 %[[IS_SOME]], i1 true)
    // OPT: br i1 %[[LIKELY]], label %[[BLOCK:.+]],
    // DBG: br i1 %[[IS_SOME]], label %[[BLOCK:.+]],

    // CHECK: [[BLOCK]]:
    // OPT: %[[SHIFTED:.+]] = sub i8 %x, -1
    // OPT: %[[IN_WIDTH:.+]] = icmp ule i8 %[[SHIFTED]], 3
    // OPT: call void @llvm.assume(i1 %[[IN_WIDTH]])
    // DBG-NOT: assume
    // CHECK: ret i8 %x

    if let Some(val) = x { val } else { unreachable!() }
}

#[no_mangle]
fn use_result_nzusize(x: Result<NonZero<usize>, NonNull<u32>>) -> NonZero<usize> {
    // CHECK-LABEL: @use_result_nzusize
    // OPT-SAME: (i64 noundef range(i64 0, 2) %x.0, ptr noundef %x.1)

    // CHECK-NOT: alloca
    // CHECK: %[[IS_ERR:.+]] = trunc nuw i64 %x.0 to i1
    // OPT: %[[UNLIKELY:.+]] = call i1 @llvm.expect.i1(i1 %[[IS_ERR]], i1 false)
    // OPT: br i1 %[[UNLIKELY]], label %[[PANIC:.+]], label %[[BLOCK:.+]]
    // DBG: br i1 %[[IS_ERR]], label %[[PANIC:.+]], label %[[BLOCK:.+]]

    // CHECK: [[BLOCK]]:
    // CHECK: %val = ptrtoint ptr %x.1 to i64
    // CHECK: ret i64 %val

    if let Ok(val) = x { val } else { unreachable!() }
}

#[no_mangle]
fn use_result_i32_char(x: Result<i32, char>) -> char {
    // CHECK-LABEL: @use_result_i32_char
    // OPT-SAME: (i32 noundef range(i32 0, 2) %x.0, i32 noundef %x.1)

    // CHECK-NOT: alloca
    // CHECK: %[[IS_ERR_WIDE:.+]] = zext i32 %x.0 to i64
    // CHECK: %[[IS_ERR:.+]] = trunc nuw i64 %[[IS_ERR_WIDE]] to i1
    // OPT: %[[LIKELY:.+]] = call i1 @llvm.expect.i1(i1 %[[IS_ERR]], i1 true)
    // OPT: br i1 %[[LIKELY]], label %[[BLOCK:.+]], label %[[PANIC:.+]]
    // DBG: br i1 %[[IS_ERR]], label %[[BLOCK:.+]], label %[[PANIC:.+]]

    // CHECK: [[BLOCK]]:
    // OPT: %[[RANGE:.+]] = icmp ule i32 %x.1, 1114111
    // OPT: call void @llvm.assume(i1 %[[RANGE]])
    // DBG-NOT: call
    // CHECK: ret i32 %x.1

    if let Err(val) = x { val } else { unreachable!() }
}

#[repr(u64)]
enum BigEnum {
    Foo = 100,
    Bar = 200,
}

#[no_mangle]
fn use_result_bigenum(x: Result<BigEnum, u64>) -> BigEnum {
    // CHECK-LABEL: @use_result_bigenum
    // OPT-SAME: (i64 noundef range(i64 0, 2) %x.0, i64 noundef %x.1)

    // CHECK-NOT: alloca
    // CHECK: %[[IS_ERR:.+]] = trunc nuw i64 %x.0 to i1
    // OPT: %[[UNLIKELY:.+]] = call i1 @llvm.expect.i1(i1 %[[IS_ERR]], i1 false)
    // OPT: br i1 %[[UNLIKELY]], label %[[PANIC:.+]], label %[[BLOCK:.+]]
    // DBG: br i1 %[[IS_ERR]], label %[[PANIC:.+]], label %[[BLOCK:.+]]

    // CHECK: [[BLOCK]]:
    // CHECK: ret i64 %x.1

    if let Ok(val) = x { val } else { unreachable!() }
}

struct WhateverError;

#[no_mangle]
fn use_result_nonnull(x: Result<NonNull<u16>, WhateverError>) -> NonNull<u16> {
    // CHECK-LABEL: @use_result_nonnull
    // OPT-SAME: (ptr noundef %x)

    // CHECK-NOT: alloca
    // CHECK: %[[ADDR:.+]] = ptrtoint ptr %x to i64
    // CHECK: %[[IS_NULL:.+]] = icmp eq i64 %[[ADDR]], 0
    // CHECK: %[[DISCR:.+]] = select i1 %[[IS_NULL]], i64 1, i64 0
    // CHECK: %[[IS_ERR:.+]] = trunc nuw i64 %[[DISCR]] to i1
    // OPT: %[[UNLIKELY:.+]] = call i1 @llvm.expect.i1(i1 %[[IS_ERR]], i1 false)
    // OPT: br i1 %[[UNLIKELY]], label %[[PANIC:.+]], label %[[BLOCK:.+]]
    // DBG: br i1 %[[IS_ERR]], label %[[PANIC:.+]], label %[[BLOCK:.+]]

    // CHECK: [[BLOCK]]:
    // CHECK: ret ptr %x

    if let Ok(val) = x { val } else { unreachable!() }
}
