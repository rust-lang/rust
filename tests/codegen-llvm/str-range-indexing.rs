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

// CHECK-LABEL: @get_range
// CHECK-COUNT-9: %{{.+}} = icmp
// CHECK-NOT: %{{.+}} = icmp
// CHECK: ret

// CHECK-LABEL: @index_range
// CHECK-COUNT-9: %{{.+}} = icmp
// CHECK-NOT: %{{.+}} = icmp
// CHECK: ret
tests!(Range<usize>, get_range, index_range);

// CHECK-LABEL: @get_range_to
// CHECK-COUNT-4: %{{.+}} = icmp
// CHECK-NOT: %{{.+}} = icmp
// CHECK: ret

// CHECK-LABEL: @index_range_to
// CHECK-COUNT-4: %{{.+}} = icmp
// CHECK-NOT: %{{.+}} = icmp
// CHECK: ret
tests!(RangeTo<usize>, get_range_to, index_range_to);

// CHECK-LABEL: @get_range_from
// CHECK-COUNT-4: %{{.+}} = icmp
// CHECK-NOT: %{{.+}} = icmp
// CHECK: ret

// CHECK-LABEL: @index_range_from
// CHECK-COUNT-4: %{{.+}} = icmp
// CHECK-NOT: %{{.+}} = icmp
// CHECK: ret
tests!(RangeFrom<usize>, get_range_from, index_range_from);

// CHECK-LABEL: @get_range_inclusive
// CHECK-COUNT-7: %{{.+}} = icmp
// CHECK-NOT: %{{.+}} = icmp
// CHECK: ret

// CHECK-LABEL: @index_range_inclusive
// CHECK-COUNT-7: %{{.+}} = icmp
// CHECK-NOT: %{{.+}} = icmp
// CHECK: ret
tests!(RangeInclusive<usize>, get_range_inclusive, index_range_inclusive);

// CHECK-LABEL: @get_range_to_inclusive
// CHECK-COUNT-3: %{{.+}} = icmp
// CHECK-NOT: %{{.+}} = icmp
// CHECK: ret

// CHECK-LABEL: @index_range_to_inclusive
// CHECK-COUNT-3: %{{.+}} = icmp
// CHECK-NOT: %{{.+}} = icmp
// CHECK: ret
tests!(RangeToInclusive<usize>, get_range_to_inclusive, index_range_to_inclusive);
