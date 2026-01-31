//@ compile-flags: -Copt-level=3
//@ min-llvm-version: 21

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

// 2 comparisons required: (range.end < slice.len()) && (range.start <= range.end)
// CHECK-LABEL: @get_range
// CHECK-COUNT-2: %{{.+}} = icmp
// CHECK-NOT: %{{.+}} = icmp
// CHECK: ret

// 2 comparisons required: (range.end < slice.len()) && (range.start <= range.end)
// CHECK-LABEL: @index_range
// CHECK-COUNT-2: %{{.+}} = icmp
// CHECK-NOT: %{{.+}} = icmp
// CHECK: ret
tests!(Range<usize>, get_range, index_range);

// 1 comparison required: (range.end < slice.len())
// CHECK-LABEL: @get_range_to
// CHECK-COUNT-1: %{{.+}} = icmp
// CHECK-NOT: %{{.+}} = icmp
// CHECK: ret

// 1 comparison required: (range.end < slice.len())
// CHECK-LABEL: @index_range_to
// CHECK-COUNT-1: %{{.+}} = icmp
// CHECK-NOT: %{{.+}} = icmp
// CHECK: ret
tests!(RangeTo<usize>, get_range_to, index_range_to);

// 1 comparison required: (range.start <= slice.len())
// CHECK-LABEL: @get_range_from
// CHECK-COUNT-1: %{{.+}} = icmp
// CHECK-NOT: %{{.+}} = icmp
// CHECK: ret

// 1 comparison required: (range.start <= slice.len())
// CHECK-LABEL: @index_range_from
// CHECK-COUNT-1: %{{.+}} = icmp
// CHECK-NOT: %{{.+}} = icmp
// CHECK: ret
tests!(RangeFrom<usize>, get_range_from, index_range_from);

// 2 comparisons required: (range.end < slice.len()) && (range.start <= range.end + 1)
// CHECK-LABEL: @get_range_inclusive
// CHECK-COUNT-2: %{{.+}} = icmp
// CHECK-NOT: %{{.+}} = icmp
// CHECK: ret

// 2 comparisons required: (range.end < slice.len()) && (range.start <= range.end + 1)
// CHECK-LABEL: @index_range_inclusive
// CHECK-COUNT-2: %{{.+}} = icmp
// CHECK-NOT: %{{.+}} = icmp
// CHECK: ret
tests!(RangeInclusive<usize>, get_range_inclusive, index_range_inclusive);

// 1 comparison required: (range.end < slice.len())
// CHECK-LABEL: @get_range_to_inclusive
// CHECK-COUNT-1: %{{.+}} = icmp
// CHECK-NOT: %{{.+}} = icmp
// CHECK: ret

// 1 comparison required: (range.end < slice.len())
// CHECK-LABEL: @index_range_to_inclusive
// CHECK-COUNT-1: %{{.+}} = icmp
// CHECK-NOT: %{{.+}} = icmp
// CHECK: ret
tests!(RangeToInclusive<usize>, get_range_to_inclusive, index_range_to_inclusive);
