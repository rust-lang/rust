//@ compile-flags: -Copt-level=3

#![crate_type = "lib"]

use std::ops::{Range, RangeFrom, RangeInclusive, RangeTo, RangeToInclusive};

macro_rules! tests {
    ($range_ty:ty, $get_func_name:ident, $index_func_name:ident) => {
        #[no_mangle]
        pub fn $get_func_name(slice: &[u32], range: $range_ty) -> Option<&[u32]> {
            slice.get(range)
        }

        #[no_mangle]
        pub fn $index_func_name(slice: &[u32], range: $range_ty) -> &[u32] {
            &slice[range]
        }
    };
}

// CHECK-LABEL: @get_range
// CHECK-COUNT-2: icmp
// 2 comparisons required: (range.end < slice.len()) && (range.start <= range.end)
// CHECK-LABEL: @index_range
// CHECK-COUNT-2: icmp
// 2 comparisons required: (range.end < slice.len()) && (range.start <= range.end)
tests!(Range<usize>, get_range, index_range);

// CHECK-LABEL: @get_range_to
// CHECK-COUNT-1: icmp
// 1 comparison required: (range.end < slice.len())
// CHECK-LABEL: @index_range_to
// CHECK-COUNT-1: icmp
// 1 comparison required: (range.end < slice.len())
tests!(RangeTo<usize>, get_range_to, index_range_to);

// CHECK-LABEL: @get_range_from
// CHECK-COUNT-1: icmp
// 1 comparison required: (range.start <= slice.len())
// CHECK-LABEL: @index_range_from
// CHECK-COUNT-1: icmp
// 1 comparison required: (range.start <= slice.len())
tests!(RangeFrom<usize>, get_range_from, index_range_from);

// CHECK-LABEL: @get_range_inclusive
// CHECK-COUNT-2: icmp
// 2 comparisons required: (range.end < slice.len()) && (range.start <= range.end + 1)
// CHECK-LABEL: @index_range_inclusive
// CHECK-COUNT-2: icmp
// 2 comparisons required: (range.end < slice.len()) && (range.start <= range.end + 1)
tests!(RangeInclusive<usize>, get_range_inclusive, index_range_inclusive);

// CHECK-LABEL: @get_range_to_inclusive
// CHECK-COUNT-1: icmp
// 1 comparison required: (range.end < slice.len())
// CHECK-LABEL: @index_range_to_inclusive
// CHECK-COUNT-1: icmp
// 1 comparison required: (range.end < slice.len())
tests!(RangeToInclusive<usize>, get_range_to_inclusive, index_range_to_inclusive);
