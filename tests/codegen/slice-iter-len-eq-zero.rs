//@ compile-flags: -Copt-level=3
//@ only-64bit
//@ needs-deterministic-layouts (opposite scalar pair orders breaks it)
#![crate_type = "lib"]

type Demo = [u8; 3];

// CHECK-LABEL: @slice_iter_len_eq_zero
#[no_mangle]
pub fn slice_iter_len_eq_zero(y: std::slice::Iter<'_, Demo>) -> bool {
    // CHECK: start:
    // CHECK: %[[NOT_ZERO:.+]] = icmp ne i64 %1, 0
    // CHECK: call void @llvm.assume(i1 %[[NOT_ZERO]])
    // CHECK: %[[PTR_ADDR:.+]] = ptrtoint ptr %0 to i64
    // CHECK: %[[RET:.+]] = icmp eq i64 %1, %[[PTR_ADDR]]
    // CHECK: ret i1 %[[RET]]
    y.len() == 0
}

// CHECK-LABEL: @slice_iter_len_eq_zero_ref
#[no_mangle]
pub fn slice_iter_len_eq_zero_ref(y: &mut std::slice::Iter<'_, Demo>) -> bool {
    // CHECK-NOT: sub
    // CHECK: %[[A:.+]] = load ptr
    // CHECK-SAME: !nonnull
    // CHECK: %[[B:.+]] = load ptr
    // CHECK-SAME: !nonnull
    // CHECK: %[[RET:.+]] = icmp eq ptr %[[B]], %[[A]]
    // CHECK: ret i1 %[[RET]]
    y.len() == 0
}

struct MyZST;

// CHECK-LABEL: @slice_zst_iter_len_eq_zero
#[no_mangle]
pub fn slice_zst_iter_len_eq_zero(y: std::slice::Iter<'_, MyZST>) -> bool {
    // CHECK: %[[RET:.+]] = icmp eq i64 %y.1, 0
    // CHECK: ret i1 %[[RET]]
    y.len() == 0
}

// CHECK-LABEL: @slice_zst_iter_len_eq_zero_ref
#[no_mangle]
pub fn slice_zst_iter_len_eq_zero_ref(y: &mut std::slice::Iter<'_, MyZST>) -> bool {
    // CHECK: %[[LEN:.+]] = load i64
    // CHECK-NOT: !range
    // CHECK: %[[RET:.+]] = icmp eq i64 %[[LEN]], 0
    // CHECK: ret i1 %[[RET]]
    y.len() == 0
}

// CHECK-LABEL: @array_into_iter_len_eq_zero
#[no_mangle]
pub fn array_into_iter_len_eq_zero(y: std::array::IntoIter<Demo, 123>) -> bool {
    // This should be able to just check that the indexes are equal, and not
    // need any subtractions or comparisons to handle `start > end`.

    // CHECK-NOT: icmp
    // CHECK-NOT: sub
    // CHECK: %_0 = icmp eq {{i16|i32|i64}}
    // CHECK: ret i1 %_0
    y.len() == 0
}
