//@ compile-flags: -O -Zmerge-functions=disabled
#![crate_type = "lib"]

use std::collections::Bound;
use std::ops::RangeBounds;

#[no_mangle]
pub fn raw_index(buf: &[u8]) -> Option<&[u8]> {
    // CHECK-LABEL: @raw_index(
    // CHECK-NOT: slice_index_fail
    // CHECK-NOT: br {{.*}}
    if buf.len() < 4 { None } else { Some(&buf[4..]) }
}

#[no_mangle]
pub fn bounds_indexer(buf: &[u8]) -> Option<&[u8]> {
    // CHECK-LABEL: @bounds_indexer(
    // CHECK-NOT: slice_index_fail
    // CHECK-NOT: br {{.*}}
    // CHECK: ret
    if buf.len() < 4 { None } else { Some(&buf[(Bound::Included(4), Bound::Unbounded)]) }
}

#[no_mangle]
pub fn bounds_indexer_with_range_bounds(buf: &[u8]) -> Option<&[u8]> {
    // CHECK-LABEL: @bounds_indexer_with_range_bounds(
    // CHECK-NOT: slice_index_fail
    // CHECK-NOT: br {{.*}}
    // CHECK: ret
    fn index(buf: &[u8], range: impl RangeBounds<usize>) -> &[u8] {
        &buf[(range.start_bound().map(|x| *x), range.end_bound().map(|x| *x))]
    }

    if buf.len() < 4 { None } else { Some(index(buf, 4..)) }
}

#[no_mangle]
pub fn bounds_indexer_with_range_bounds_manual_map(buf: &[u8]) -> Option<&[u8]> {
    // CHECK-LABEL: @bounds_indexer_with_range_bounds_manual_map(
    // CHECK-NOT: slice_index_fail
    // CHECK-NOT: br {{.*}}
    // CHECK: ret
    fn index(buf: &[u8], range: impl RangeBounds<usize>) -> &[u8] {
        &buf[(
            match range.start_bound() {
                Bound::Included(&i) => Bound::Included(i),
                Bound::Excluded(&i) => Bound::Excluded(i),
                Bound::Unbounded => Bound::Unbounded,
            },
            match range.end_bound() {
                Bound::Included(&i) => Bound::Included(i),
                Bound::Excluded(&i) => Bound::Excluded(i),
                Bound::Unbounded => Bound::Unbounded,
            },
        )]
    }

    if buf.len() < 4 { None } else { Some(index(buf, 4..)) }
}

#[no_mangle]
pub fn bounds_indexer_with_range_bounds_manually_mapped(buf: &[u8]) -> Option<&[u8]> {
    // CHECK-LABEL: @bounds_indexer_with_range_bounds_manually_mapped(
    // CHECK-NOT: slice_index_fail
    // CHECK-NOT: br {{.*}}
    // CHECK: ret
    fn index(buf: &[u8], range: impl RangeBounds<usize>) -> &[u8] {
        &buf[match range.start_bound() {
            Bound::Included(&i) => i,
            Bound::Excluded(i) => i.checked_add(1).expect("overflow"),
            Bound::Unbounded => 0,
        }..match range.end_bound() {
            Bound::Included(&i) => i,
            Bound::Excluded(i) => i.checked_sub(1).expect("overflow"),
            Bound::Unbounded => buf.len(),
        }]
    }

    if buf.len() < 4 { None } else { Some(index(buf, 4..)) }
}
