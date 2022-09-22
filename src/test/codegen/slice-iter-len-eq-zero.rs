// no-system-llvm
// compile-flags: -O
// ignore-debug: the debug assertions add extra comparisons
#![crate_type = "lib"]

type Demo = [u8; 3];

// CHECK-LABEL: @slice_iter_len_eq_zero
#[no_mangle]
pub fn slice_iter_len_eq_zero(y: std::slice::Iter<'_, Demo>) -> bool {
    // CHECK-NOT: sub
    // CHECK: %2 = icmp eq {{i8\*|ptr}} %1, %0
    // CHECK: ret i1 %2
    y.len() == 0
}

// CHECK-LABEL: @array_into_iter_len_eq_zero
#[no_mangle]
pub fn array_into_iter_len_eq_zero(y: std::array::IntoIter<Demo, 123>) -> bool {
    // This should be able to just check that the indexes are equal, and not
    // need any subtractions or comparisons to handle `start > end`.

    // CHECK-NOT: icmp
    // CHECK-NOT: sub
    // CHECK: %1 = icmp eq {{i16|i32|i64}}
    // CHECK: ret i1 %1
    y.len() == 0
}
