//@ compile-flags: -C opt-level=3
#![crate_type = "lib"]

// These pipelines use integer elements without drop glue and reducers that cannot panic. The
// temporary allocation can therefore be removed without changing panic or drop ordering.

// CHECK-LABEL: define{{.*}} @sum_of_squares
// CHECK-NOT: __rust_alloc
#[no_mangle]
pub fn sum_of_squares(nums: &[u64]) -> u64 {
    let squares: Vec<u64> = nums.iter().map(|n| n.wrapping_mul(*n)).collect();
    squares.into_iter().fold(0, u64::wrapping_add)
}

// CHECK-LABEL: define{{.*}} @filtered_rotating_product
// CHECK-NOT: __rust_alloc
#[no_mangle]
pub fn filtered_rotating_product(nums: &[u32]) -> u32 {
    let values: Vec<u32> =
        nums.iter().filter(|n| **n != 0).copied().map(|n| n.rotate_left(5)).collect();
    values.into_iter().fold(1, u32::wrapping_mul)
}

// CHECK-LABEL: define{{.*}} @cloned_saturating_sum
// CHECK-NOT: __rust_alloc
#[no_mangle]
pub fn cloned_saturating_sum(nums: &[i16]) -> i16 {
    let values: Vec<i16> = nums.iter().cloned().collect();
    values.into_iter().fold(0, i16::saturating_add)
}

// CHECK-LABEL: define{{.*}} @enumerated_saturating_difference
// CHECK-NOT: __rust_alloc
#[no_mangle]
pub fn enumerated_saturating_difference(nums: &[usize]) -> usize {
    let values: Vec<usize> =
        nums.iter().copied().enumerate().map(|(index, n)| n.saturating_add(index)).collect();
    values.into_iter().fold(usize::MAX, usize::saturating_sub)
}

// CHECK-LABEL: define{{.*}} @aliased_wrapping_difference
// CHECK-NOT: __rust_alloc
#[no_mangle]
pub fn aliased_wrapping_difference(nums: &[u8]) -> u8 {
    let values: Vec<u8> = nums.iter().map(|n| n.reverse_bits()).collect();
    let first_alias = values;
    let second_alias = first_alias;
    second_alias.into_iter().fold(0, u8::wrapping_sub)
}

// CHECK-LABEL: define{{.*}} @saturating_product
// CHECK-NOT: __rust_alloc
#[no_mangle]
pub fn saturating_product(nums: &[i32]) -> i32 {
    let values: Vec<i32> = nums.iter().map(|n| n.saturating_add(1)).collect();
    values.into_iter().fold(1, i32::saturating_mul)
}

// A captured adapter closure is deliberately outside the recognition scope.
// CHECK-LABEL: define{{.*}} @capturing_map_is_not_elided
// CHECK: __rust_alloc
#[no_mangle]
pub fn capturing_map_is_not_elided(nums: &[u64], offset: u64) -> u64 {
    let values: Vec<u64> = nums.iter().map(|n| n.wrapping_add(offset)).collect();
    values.into_iter().fold(0, u64::wrapping_add)
}

// Observing the Vec gives the allocation more than one external use.
// CHECK-LABEL: define{{.*}} @observed_vec_is_not_elided
// CHECK: __rust_alloc
#[no_mangle]
pub fn observed_vec_is_not_elided(nums: &[u64]) -> u64 {
    let values: Vec<u64> = nums.iter().map(|n| n.wrapping_mul(*n)).collect();
    let len = values.len() as u64;
    values.into_iter().fold(0, u64::wrapping_add).wrapping_add(len)
}

// Reducer closures are not assumed to be non-panicking, even when this one happens to be.
// CHECK-LABEL: define{{.*}} @closure_reducer_is_not_elided
// CHECK: __rust_alloc
#[no_mangle]
pub fn closure_reducer_is_not_elided(nums: &[u64]) -> u64 {
    let values: Vec<u64> = nums.iter().map(|n| n.wrapping_mul(*n)).collect();
    values.into_iter().fold(0, |acc, value| acc.wrapping_add(value))
}
