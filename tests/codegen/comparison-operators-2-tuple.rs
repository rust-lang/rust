// compile-flags: -C opt-level=1 -Z merge-functions=disabled
// min-llvm-version: 15.0
// only-x86_64

#![crate_type = "lib"]

use std::cmp::Ordering;

type TwoTuple = (i16, u16);

//
// The operators are all overridden directly, so should optimize easily.
//
// Yes, the `s[lg]t` is correct for the `[lg]e` version because it's only used
// in the side of the select where we know the values are *not* equal.
//

// CHECK-LABEL: @check_lt_direct
// CHECK-SAME: (i16 noundef %[[A0:.+]], i16 noundef %[[A1:.+]], i16 noundef %[[B0:.+]], i16 noundef %[[B1:.+]])
#[no_mangle]
pub fn check_lt_direct(a: TwoTuple, b: TwoTuple) -> bool {
    // CHECK-DAG: %[[EQ:.+]] = icmp eq i16 %[[A0]], %[[B0]]
    // CHECK-DAG: %[[CMP0:.+]] = icmp slt i16 %[[A0]], %[[B0]]
    // CHECK-DAG: %[[CMP1:.+]] = icmp ult i16 %[[A1]], %[[B1]]
    // CHECK: %[[R:.+]] = select i1 %[[EQ]], i1 %[[CMP1]], i1 %[[CMP0]]
    // CHECK: ret i1 %[[R]]
    a < b
}

// CHECK-LABEL: @check_le_direct
// CHECK-SAME: (i16 noundef %[[A0:.+]], i16 noundef %[[A1:.+]], i16 noundef %[[B0:.+]], i16 noundef %[[B1:.+]])
#[no_mangle]
pub fn check_le_direct(a: TwoTuple, b: TwoTuple) -> bool {
    // CHECK-DAG: %[[EQ:.+]] = icmp eq i16 %[[A0]], %[[B0]]
    // CHECK-DAG: %[[CMP0:.+]] = icmp slt i16 %[[A0]], %[[B0]]
    // CHECK-DAG: %[[CMP1:.+]] = icmp ule i16 %[[A1]], %[[B1]]
    // CHECK: %[[R:.+]] = select i1 %[[EQ]], i1 %[[CMP1]], i1 %[[CMP0]]
    // CHECK: ret i1 %[[R]]
    a <= b
}

// CHECK-LABEL: @check_gt_direct
// CHECK-SAME: (i16 noundef %[[A0:.+]], i16 noundef %[[A1:.+]], i16 noundef %[[B0:.+]], i16 noundef %[[B1:.+]])
#[no_mangle]
pub fn check_gt_direct(a: TwoTuple, b: TwoTuple) -> bool {
    // CHECK-DAG: %[[EQ:.+]] = icmp eq i16 %[[A0]], %[[B0]]
    // CHECK-DAG: %[[CMP0:.+]] = icmp sgt i16 %[[A0]], %[[B0]]
    // CHECK-DAG: %[[CMP1:.+]] = icmp ugt i16 %[[A1]], %[[B1]]
    // CHECK: %[[R:.+]] = select i1 %[[EQ]], i1 %[[CMP1]], i1 %[[CMP0]]
    // CHECK: ret i1 %[[R]]
    a > b
}

// CHECK-LABEL: @check_ge_direct
// CHECK-SAME: (i16 noundef %[[A0:.+]], i16 noundef %[[A1:.+]], i16 noundef %[[B0:.+]], i16 noundef %[[B1:.+]])
#[no_mangle]
pub fn check_ge_direct(a: TwoTuple, b: TwoTuple) -> bool {
    // CHECK-DAG: %[[EQ:.+]] = icmp eq i16 %[[A0]], %[[B0]]
    // CHECK-DAG: %[[CMP0:.+]] = icmp sgt i16 %[[A0]], %[[B0]]
    // CHECK-DAG: %[[CMP1:.+]] = icmp uge i16 %[[A1]], %[[B1]]
    // CHECK: %[[R:.+]] = select i1 %[[EQ]], i1 %[[CMP1]], i1 %[[CMP0]]
    // CHECK: ret i1 %[[R]]
    a >= b
}

//
// These ones are harder, since there are more intermediate values to remove.
//
// `<` seems to be getting lucky right now, so test that doesn't regress.
//
// The others, however, aren't managing to optimize away the extra `select`s yet.
// See <https://github.com/rust-lang/rust/issues/106107> for more about this.
//

// CHECK-LABEL: @check_lt_via_cmp
// CHECK-SAME: (i16 noundef %[[A0:.+]], i16 noundef %[[A1:.+]], i16 noundef %[[B0:.+]], i16 noundef %[[B1:.+]])
#[no_mangle]
pub fn check_lt_via_cmp(a: TwoTuple, b: TwoTuple) -> bool {
    // CHECK-DAG: %[[EQ:.+]] = icmp eq i16 %[[A0]], %[[B0]]
    // CHECK-DAG: %[[CMP0:.+]] = icmp slt i16 %[[A0]], %[[B0]]
    // CHECK-DAG: %[[CMP1:.+]] = icmp ult i16 %[[A1]], %[[B1]]
    // CHECK: %[[R:.+]] = select i1 %[[EQ]], i1 %[[CMP1]], i1 %[[CMP0]]
    // CHECK: ret i1 %[[R]]
    Ord::cmp(&a, &b).is_lt()
}

// CHECK-LABEL: @check_le_via_cmp
// CHECK-SAME: (i16 noundef %[[A0:.+]], i16 noundef %[[A1:.+]], i16 noundef %[[B0:.+]], i16 noundef %[[B1:.+]])
#[no_mangle]
pub fn check_le_via_cmp(a: TwoTuple, b: TwoTuple) -> bool {
    // FIXME-CHECK-DAG: %[[EQ:.+]] = icmp eq i16 %[[A0]], %[[B0]]
    // FIXME-CHECK-DAG: %[[CMP0:.+]] = icmp sle i16 %[[A0]], %[[B0]]
    // FIXME-CHECK-DAG: %[[CMP1:.+]] = icmp ule i16 %[[A1]], %[[B1]]
    // FIXME-CHECK: %[[R:.+]] = select i1 %[[EQ]], i1 %[[CMP1]], i1 %[[CMP0]]
    // FIXME-CHECK: ret i1 %[[R]]
    Ord::cmp(&a, &b).is_le()
}

// CHECK-LABEL: @check_gt_via_cmp
// CHECK-SAME: (i16 noundef %[[A0:.+]], i16 noundef %[[A1:.+]], i16 noundef %[[B0:.+]], i16 noundef %[[B1:.+]])
#[no_mangle]
pub fn check_gt_via_cmp(a: TwoTuple, b: TwoTuple) -> bool {
    // FIXME-CHECK-DAG: %[[EQ:.+]] = icmp eq i16 %[[A0]], %[[B0]]
    // FIXME-CHECK-DAG: %[[CMP0:.+]] = icmp sgt i16 %[[A0]], %[[B0]]
    // FIXME-CHECK-DAG: %[[CMP1:.+]] = icmp ugt i16 %[[A1]], %[[B1]]
    // FIXME-CHECK: %[[R:.+]] = select i1 %[[EQ]], i1 %[[CMP1]], i1 %[[CMP0]]
    // FIXME-CHECK: ret i1 %[[R]]
    Ord::cmp(&a, &b).is_gt()
}

// CHECK-LABEL: @check_ge_via_cmp
// CHECK-SAME: (i16 noundef %[[A0:.+]], i16 noundef %[[A1:.+]], i16 noundef %[[B0:.+]], i16 noundef %[[B1:.+]])
#[no_mangle]
pub fn check_ge_via_cmp(a: TwoTuple, b: TwoTuple) -> bool {
    // FIXME-CHECK-DAG: %[[EQ:.+]] = icmp eq i16 %[[A0]], %[[B0]]
    // FIXME-CHECK-DAG: %[[CMP0:.+]] = icmp sge i16 %[[A0]], %[[B0]]
    // FIXME-CHECK-DAG: %[[CMP1:.+]] = icmp uge i16 %[[A1]], %[[B1]]
    // FIXME-CHECK: %[[R:.+]] = select i1 %[[EQ]], i1 %[[CMP1]], i1 %[[CMP0]]
    // FIXME-CHECK: ret i1 %[[R]]
    Ord::cmp(&a, &b).is_ge()
}
