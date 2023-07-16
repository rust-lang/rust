// compile-flags: -O -Zmerge-functions=disabled
// ignore-debug (the extra assertions get in the way)

#![crate_type = "lib"]

use std::num::{NonZeroI16, NonZeroU32};

// #71602 reported a simple array comparison just generating a loop.
// This was originally fixed by ensuring it generates a single bcmp,
// but we now generate it as a load+icmp instead. `is_zero_slice` was
// tweaked to still test the case of comparison against a slice,
// and `is_zero_array` tests the new array-specific behaviour.
// The optimization was then extended to short slice-to-array comparisons,
// so the first test here now has a long slice to still get the bcmp.

// CHECK-LABEL: @is_zero_slice_long
#[no_mangle]
pub fn is_zero_slice_long(data: &[u8; 456]) -> bool {
    // CHECK: %[[BCMP:.+]] = tail call i32 @{{bcmp|memcmp}}({{.+}})
    // CHECK-NEXT: %[[EQ:.+]] = icmp eq i32 %[[BCMP]], 0
    // CHECK-NEXT: ret i1 %[[EQ]]
    &data[..] == [0; 456]
}

// CHECK-LABEL: @is_zero_slice_short
#[no_mangle]
pub fn is_zero_slice_short(data: &[u8; 4]) -> bool {
    // CHECK: %[[LOAD:.+]] = load i32, {{i32\*|ptr}} %{{.+}}, align 1
    // CHECK-NEXT: %[[EQ:.+]] = icmp eq i32 %[[LOAD]], 0
    // CHECK-NEXT: ret i1 %[[EQ]]
    &data[..] == [0; 4]
}

// CHECK-LABEL: @is_zero_array
#[no_mangle]
pub fn is_zero_array(data: &[u8; 4]) -> bool {
    // CHECK: %[[LOAD:.+]] = load i32, {{i32\*|ptr}} %{{.+}}, align 1
    // CHECK-NEXT: %[[EQ:.+]] = icmp eq i32 %[[LOAD]], 0
    // CHECK-NEXT: ret i1 %[[EQ]]
    *data == [0; 4]
}

// The following test the extra specializations to make sure that slice
// equality for non-byte types also just emit a `bcmp`, not a loop.

// CHECK-LABEL: @eq_slice_of_nested_u8(
// CHECK-SAME: [[USIZE:i16|i32|i64]] noundef %x.1
// CHECK-SAME: [[USIZE]] noundef %y.1
#[no_mangle]
fn eq_slice_of_nested_u8(x: &[[u8; 3]], y: &[[u8; 3]]) -> bool {
    // CHECK: icmp eq [[USIZE]] %x.1, %y.1
    // CHECK: %[[BYTES:.+]] = mul nsw [[USIZE]] %x.1, 3
    // CHECK: tail call{{( noundef)?}} i32 @{{bcmp|memcmp}}({{i8\*|ptr}}
    // CHECK-SAME: , [[USIZE]]{{( noundef)?}} %[[BYTES]])
    x == y
}

// CHECK-LABEL: @eq_slice_of_i32(
// CHECK-SAME: [[USIZE:i16|i32|i64]] noundef %x.1
// CHECK-SAME: [[USIZE]] noundef %y.1
#[no_mangle]
fn eq_slice_of_i32(x: &[i32], y: &[i32]) -> bool {
    // CHECK: icmp eq [[USIZE]] %x.1, %y.1
    // CHECK: %[[BYTES:.+]] = shl nsw [[USIZE]] %x.1, 2
    // CHECK: tail call{{( noundef)?}} i32 @{{bcmp|memcmp}}({{i32\*|ptr}}
    // CHECK-SAME: , [[USIZE]]{{( noundef)?}} %[[BYTES]])
    x == y
}

// CHECK-LABEL: @eq_slice_of_nonzero(
// CHECK-SAME: [[USIZE:i16|i32|i64]] noundef %x.1
// CHECK-SAME: [[USIZE]] noundef %y.1
#[no_mangle]
fn eq_slice_of_nonzero(x: &[NonZeroU32], y: &[NonZeroU32]) -> bool {
    // CHECK: icmp eq [[USIZE]] %x.1, %y.1
    // CHECK: %[[BYTES:.+]] = shl nsw [[USIZE]] %x.1, 2
    // CHECK: tail call{{( noundef)?}} i32 @{{bcmp|memcmp}}({{i32\*|ptr}}
    // CHECK-SAME: , [[USIZE]]{{( noundef)?}} %[[BYTES]])
    x == y
}

// CHECK-LABEL: @eq_slice_of_option_of_nonzero(
// CHECK-SAME: [[USIZE:i16|i32|i64]] noundef %x.1
// CHECK-SAME: [[USIZE]] noundef %y.1
#[no_mangle]
fn eq_slice_of_option_of_nonzero(x: &[Option<NonZeroI16>], y: &[Option<NonZeroI16>]) -> bool {
    // CHECK: icmp eq [[USIZE]] %x.1, %y.1
    // CHECK: %[[BYTES:.+]] = shl nsw [[USIZE]] %x.1, 1
    // CHECK: tail call{{( noundef)?}} i32 @{{bcmp|memcmp}}({{i16\*|ptr}}
    // CHECK-SAME: , [[USIZE]]{{( noundef)?}} %[[BYTES]])
    x == y
}
