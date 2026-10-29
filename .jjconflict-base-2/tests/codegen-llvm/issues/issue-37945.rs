//@ compile-flags: -Copt-level=3 -Zmerge-functions=disabled
//@ ignore-32bit LLVM has a bug with them
//@ revisions: new old
//@ [old] max-llvm-major-version: 22
//@ [new] min-llvm-version: 23

// Check that LLVM understands that `Iter` pointer is not null. Issue #37945.

#![crate_type = "lib"]

use std::slice::Iter;

#[no_mangle]
pub fn is_empty_1(xs: Iter<f32>) -> bool {
    // CHECK-LABEL: @is_empty_1(
    // CHECK-NEXT:  start:
    // old-NEXT:    [[A:%.*]] = icmp ne ptr {{%xs.0|%xs.1}}, null
    // old-NEXT:    tail call void @llvm.assume(i1 [[A]])
    // new-NEXT:    call void @llvm.assume(i1 true) [ "nonnull"(ptr {{%xs.0|%xs.1}}) ]
    // The order between %xs.0 and %xs.1 on the next line doesn't matter
    // and different LLVM versions produce different order.
    // CHECK-NEXT:    [[B:%.*]] = icmp eq ptr {{%xs.0, %xs.1|%xs.1, %xs.0}}
    // CHECK-NEXT:    ret i1 [[B:%.*]]
    { xs }.next().is_none()
}

#[no_mangle]
pub fn is_empty_2(xs: Iter<f32>) -> bool {
    // CHECK-LABEL: @is_empty_2
    // CHECK-NEXT:  start:
    // old-NEXT:    [[C:%.*]] = icmp ne ptr {{%xs.0|%xs.1}}, null
    // old-NEXT:    tail call void @llvm.assume(i1 [[C]])
    // new-NEXT:    call void @llvm.assume(i1 true) [ "nonnull"(ptr {{%xs.0|%xs.1}}) ]
    // The order between %xs.0 and %xs.1 on the next line doesn't matter
    // and different LLVM versions produce different order.
    // CHECK-NEXT:    [[D:%.*]] = icmp eq ptr {{%xs.0, %xs.1|%xs.1, %xs.0}}
    // CHECK-NEXT:    ret i1 [[D:%.*]]
    xs.map(|&x| x).next().is_none()
}
