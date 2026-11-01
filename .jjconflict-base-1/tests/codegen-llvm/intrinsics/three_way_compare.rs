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
    // CHECK: %[[CMP:.+]] = call i8 @llvm.scmp.i8.i16(i16 %a, i16 %b)
    // CHECK-NEXT: ret i8 %[[CMP]]
    three_way_compare(a, b)
}

#[no_mangle]
// CHECK-LABEL: @unsigned_cmp
// CHECK-SAME: (i16{{.*}} %a, i16{{.*}} %b)
pub fn unsigned_cmp(a: u16, b: u16) -> std::cmp::Ordering {
    // CHECK: %[[CMP:.+]] = call i8 @llvm.ucmp.i8.i16(i16 %a, i16 %b)
    // CHECK-NEXT: ret i8 %[[CMP]]
    three_way_compare(a, b)
}
