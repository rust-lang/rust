//@ revisions: DEBUG OPTIM
//@ [DEBUG] compile-flags: -C opt-level=0
//@ [OPTIM] compile-flags: -C opt-level=3
//@ compile-flags: -C no-prepopulate-passes

#![crate_type = "lib"]
#![feature(core_intrinsics)]

use std::intrinsics::three_way_compare;

#[no_mangle]
// CHECK-LABEL: @signed_cmp
// CHECK-SAME: (i16{{.*}} %a, i16{{.*}} %b)
pub fn signed_cmp(a: i16, b: i16) -> std::cmp::Ordering {
    // DEBUG: %[[GT:.+]] = icmp sgt i16 %a, %b
    // DEBUG: %[[ZGT:.+]] = zext i1 %[[GT]] to i8
    // DEBUG: %[[LT:.+]] = icmp slt i16 %a, %b
    // DEBUG: %[[ZLT:.+]] = zext i1 %[[LT]] to i8
    // DEBUG: %[[R:.+]] = sub nsw i8 %[[ZGT]], %[[ZLT]]

    // OPTIM: %[[LT:.+]] = icmp slt i16 %a, %b
    // OPTIM: %[[NE:.+]] = icmp ne i16 %a, %b
    // OPTIM: %[[CGE:.+]] = select i1 %[[NE]], i8 1, i8 0
    // OPTIM: %[[CGEL:.+]] = select i1 %[[LT]], i8 -1, i8 %[[CGE]]
    // OPTIM: ret i8 %[[CGEL]]
    three_way_compare(a, b)
}

#[no_mangle]
// CHECK-LABEL: @unsigned_cmp
// CHECK-SAME: (i16{{.*}} %a, i16{{.*}} %b)
pub fn unsigned_cmp(a: u16, b: u16) -> std::cmp::Ordering {
    // DEBUG: %[[GT:.+]] = icmp ugt i16 %a, %b
    // DEBUG: %[[ZGT:.+]] = zext i1 %[[GT]] to i8
    // DEBUG: %[[LT:.+]] = icmp ult i16 %a, %b
    // DEBUG: %[[ZLT:.+]] = zext i1 %[[LT]] to i8
    // DEBUG: %[[R:.+]] = sub nsw i8 %[[ZGT]], %[[ZLT]]

    // OPTIM: %[[LT:.+]] = icmp ult i16 %a, %b
    // OPTIM: %[[NE:.+]] = icmp ne i16 %a, %b
    // OPTIM: %[[CGE:.+]] = select i1 %[[NE]], i8 1, i8 0
    // OPTIM: %[[CGEL:.+]] = select i1 %[[LT]], i8 -1, i8 %[[CGE]]
    // OPTIM: ret i8 %[[CGEL]]
    three_way_compare(a, b)
}
