//@ compile-flags: -Copt-level=3
//@ min-llvm-version: 21

#![crate_type = "lib"]

use std::ops::{Range, RangeFrom, RangeInclusive, RangeTo, RangeToInclusive};

macro_rules! tests {
    ($range_ty:ty, $get_func_name:ident, $index_func_name:ident) => {
        #[no_mangle]
        pub fn $get_func_name(slice: &str, range: $range_ty) -> Option<&str> {
            slice.get(range)
        }

        #[no_mangle]
        pub fn $index_func_name(slice: &str, range: $range_ty) -> &str {
            &slice[range]
        }
    };
}

// 9 comparisons required:
// start <= end
// && (start == 0 || (start >= len && start == len) || bytes[start] >= -0x40)
// && (end   == 0 || (end   >= len && end   == len) || bytes[end]   >= -0x40)

// CHECK-LABEL: @get_range
// CHECK-COUNT-9: %{{.+}} = icmp
// CHECK-NOT: %{{.+}} = icmp
// CHECK: ret

// CHECK-LABEL: @index_range
// CHECK-COUNT-9: %{{.+}} = icmp
// CHECK-NOT: %{{.+}} = icmp
// CHECK: ret
tests!(Range<usize>, get_range, index_range);

// 7 comparisons required:
// end < len && start <= end + 1
// && (start == 0 || start   >= len || bytes[start]   >= -0x40)
// && (              end + 1 >= len || bytes[end + 1] >= -0x40)

// CHECK-LABEL: @get_range_inclusive
// CHECK-COUNT-7: %{{.+}} = icmp
// CHECK-NOT: %{{.+}} = icmp
// CHECK: ret

// CHECK-LABEL: @index_range_inclusive
// CHECK-COUNT-7: %{{.+}} = icmp
// CHECK-NOT: %{{.+}} = icmp
// CHECK: ret
tests!(RangeInclusive<usize>, get_range_inclusive, index_range_inclusive);

// 4 comparisons required:
// end == 0 || (end >= len && end == len) || bytes[end] >= -0x40

// CHECK-LABEL: @get_range_to
// CHECK-COUNT-4: %{{.+}} = icmp
// CHECK-NOT: %{{.+}} = icmp
// CHECK: ret

// CHECK-LABEL: @index_range_to
// CHECK-COUNT-4: %{{.+}} = icmp
// CHECK-NOT: %{{.+}} = icmp
// CHECK: ret
tests!(RangeTo<usize>, get_range_to, index_range_to);

// 3 comparisons required:
// end < len && (end + 1 >= len || bytes[end + 1] >= -0x40)

// CHECK-LABEL: @get_range_to_inclusive
// CHECK-COUNT-3: %{{.+}} = icmp
// CHECK-NOT: %{{.+}} = icmp
// CHECK: ret

// CHECK-LABEL: @index_range_to_inclusive
// CHECK-COUNT-3: %{{.+}} = icmp
// CHECK-NOT: %{{.+}} = icmp
// CHECK: ret
tests!(RangeToInclusive<usize>, get_range_to_inclusive, index_range_to_inclusive);

// 4 comparisons required:
// start == 0 || (start >= len && start == len) || bytes[start] >= -0x40)

// CHECK-LABEL: @get_range_from
// CHECK-COUNT-4: %{{.+}} = icmp
// CHECK-NOT: %{{.+}} = icmp
// CHECK: ret

// CHECK-LABEL: @index_range_from
// CHECK-COUNT-4: %{{.+}} = icmp
// CHECK-NOT: %{{.+}} = icmp
// CHECK: ret
tests!(RangeFrom<usize>, get_range_from, index_range_from);
