// Tests output of multiple permutations of `Option::or`
//@ compile-flags: -Copt-level=3 -Zmerge-functions=disabled

#![crate_type = "lib"]

use std::num::NonZero;

// CHECK-LABEL: @or_match_u8
// CHECK-SAME: (i1{{.+}}%0, i8 %1, i1{{.+}}%optb.0, i8 %optb.1)
#[no_mangle]
pub fn or_match_u8(opta: Option<u8>, optb: Option<u8>) -> Option<u8> {
    // CHECK: start:
    // CHECK-DAG: [[A_OR_B:%.+]] = select i1 %0, i8 %1, i8 %optb.1
    // CHECK-DAG: [[IS_SOME:%.+]] = or i1 {{%0, %optb.0|%optb.0, %0}}
    // CHECK-NEXT: [[FLAG:%.+]] = insertvalue { i1, i8 } poison, i1 [[IS_SOME]], 0
    // CHECK-NEXT: [[R:%.+]] = insertvalue { i1, i8 } [[FLAG]], i8 [[A_OR_B]], 1
    // CHECK: ret { i1, i8 } [[R]]
    match opta {
        Some(x) => Some(x),
        None => optb,
    }
}

// CHECK-LABEL: @or_match_alt_u8
// CHECK-SAME: (i1{{.+}}%opta.0, i8 %opta.1, i1{{.+}}%optb.0, i8 %optb.1)
#[no_mangle]
pub fn or_match_alt_u8(opta: Option<u8>, optb: Option<u8>) -> Option<u8> {
    // CHECK: start:
    // CHECK-DAG: [[A_OR_B:%.+]] = select i1 %opta.0, i8 %opta.1, i8 %optb.1
    // CHECK-DAG: [[IS_SOME:%.+]] = or i1 {{%opta.0, %optb.0|%optb.0, %opta.0}}
    // CHECK-NEXT: [[FLAG:%.+]] = insertvalue { i1, i8 } poison, i1 [[IS_SOME]], 0
    // CHECK-NEXT: [[R:%.+]] = insertvalue { i1, i8 } [[FLAG]], i8 [[A_OR_B]], 1
    // CHECK: ret { i1, i8 } [[R]]
    match opta {
        Some(_) => opta,
        None => optb,
    }
}

// CHECK-LABEL: @option_or_u8
// CHECK-SAME: (i1{{.+}}%opta.0, i8 %opta.1, i1{{.+}}%optb.0, i8 %optb.1)
#[no_mangle]
pub fn option_or_u8(opta: Option<u8>, optb: Option<u8>) -> Option<u8> {
    // CHECK: start:
    // CHECK-DAG: [[A_OR_B:%.+]] = select i1 %opta.0, i8 %opta.1, i8 %optb.1
    // CHECK-DAG: [[IS_SOME:%.+]] = or i1 {{%opta.0, %optb.0|%optb.0, %opta.0}}
    // CHECK-NEXT: [[FLAG:%.+]] = insertvalue { i1, i8 } poison, i1 [[IS_SOME]], 0
    // CHECK-NEXT: [[R:%.+]] = insertvalue { i1, i8 } [[FLAG]], i8 [[A_OR_B]], 1
    // CHECK: ret { i1, i8 } [[R]]
    opta.or(optb)
}

// CHECK-LABEL: @if_some_u8
// CHECK-SAME: (i1{{.+}}%opta.0, i8 %opta.1, i1{{.+}}%optb.0, i8 %optb.1)
#[no_mangle]
pub fn if_some_u8(opta: Option<u8>, optb: Option<u8>) -> Option<u8> {
    // CHECK: start:
    // CHECK-DAG: [[A_OR_B:%.+]] = select i1 %opta.0, i8 %opta.1, i8 %optb.1
    // CHECK-DAG: [[IS_SOME:%.+]] = or i1 {{%opta.0, %optb.0|%optb.0, %opta.0}}
    // CHECK-NEXT: [[FLAG:%.+]] = insertvalue { i1, i8 } poison, i1 [[IS_SOME]], 0
    // CHECK-NEXT: [[R:%.+]] = insertvalue { i1, i8 } [[FLAG]], i8 [[A_OR_B]], 1
    // CHECK: ret { i1, i8 } [[R]]
    if opta.is_some() { opta } else { optb }
}

// Tests a case where an input is a type that is represented as `BackendRepr::Memory`

// CHECK-LABEL: @or_match_slice_u8
// CHECK-SAME: (i16 %0, i16 %1)
#[no_mangle]
pub fn or_match_slice_u8(opta: Option<[u8; 1]>, optb: Option<[u8; 1]>) -> Option<[u8; 1]> {
    // CHECK: start:
    // CHECK-NEXT: [[SOME_A:%.+]] = trunc i16 %0 to i1
    // CHECK-NEXT: [[R:%.+]] = select i1 [[SOME_A]], i16 %0, i16 %1
    // CHECK: ret i16 [[R]]
    match opta {
        Some(x) => Some(x),
        None => optb,
    }
}

// CHECK-LABEL: @or_match_slice_alt_u8
// CHECK-SAME: (i16 %0, i16 %1)
#[no_mangle]
pub fn or_match_slice_alt_u8(opta: Option<[u8; 1]>, optb: Option<[u8; 1]>) -> Option<[u8; 1]> {
    // CHECK: start:
    // CHECK-NEXT: [[SOME_A:%.+]] = trunc i16 %0 to i1
    // CHECK-NEXT: [[R:%.+]] = select i1 [[SOME_A]], i16 %0, i16 %1
    // CHECK: ret i16 [[R]]
    match opta {
        Some(_) => opta,
        None => optb,
    }
}

// CHECK-LABEL: @option_or_slice_u8
// CHECK-SAME: (i16 %0, i16 %1)
#[no_mangle]
pub fn option_or_slice_u8(opta: Option<[u8; 1]>, optb: Option<[u8; 1]>) -> Option<[u8; 1]> {
    // CHECK: start:
    // CHECK-NEXT: [[SOME_A:%.+]] = trunc i16 %0 to i1
    // CHECK-NEXT: [[R:%.+]] = select i1 [[SOME_A]], i16 %0, i16 %1
    // CHECK: ret i16 [[R]]
    opta.or(optb)
}

// CHECK-LABEL: @if_some_slice_u8
// CHECK-SAME: (i16 %0, i16 %1)
#[no_mangle]
pub fn if_some_slice_u8(opta: Option<[u8; 1]>, optb: Option<[u8; 1]>) -> Option<[u8; 1]> {
    // CHECK: start:
    // CHECK-NEXT: [[SOME_A:%.+]] = trunc i16 %0 to i1
    // CHECK-NEXT: [[R:%.+]] = select i1 [[SOME_A]], i16 %0, i16 %1
    // CHECK: ret i16 [[R]]
    if opta.is_some() { opta } else { optb }
}

// Test a niche optimization case of `NonZero<u8>`

// CHECK-LABEL: @or_match_nz_u8
// CHECK-SAME: (i8{{.+}}%0, i8{{.+}}%optb)
#[no_mangle]
pub fn or_match_nz_u8(opta: Option<NonZero<u8>>, optb: Option<NonZero<u8>>) -> Option<NonZero<u8>> {
    // CHECK: start:
    // CHECK-NEXT: [[NOT_A:%.+]] = icmp eq i8 %0, 0
    // CHECK-NEXT: [[R:%.+]] = select i1 [[NOT_A]], i8 %optb, i8 %0
    // CHECK: ret i8 [[R]]
    match opta {
        Some(x) => Some(x),
        None => optb,
    }
}

// CHECK-LABEL: @or_match_alt_nz_u8
// CHECK-SAME: (i8{{.+}}%opta, i8{{.+}}%optb)
#[no_mangle]
pub fn or_match_alt_nz_u8(
    opta: Option<NonZero<u8>>,
    optb: Option<NonZero<u8>>,
) -> Option<NonZero<u8>> {
    // CHECK: start:
    // CHECK-NEXT: [[NOT_A:%.+]] = icmp eq i8 %opta, 0
    // CHECK-NEXT: [[R:%.+]] = select i1 [[NOT_A]], i8 %optb, i8 %opta
    // CHECK: ret i8 [[R]]
    match opta {
        Some(_) => opta,
        None => optb,
    }
}

// CHECK-LABEL: @option_or_nz_u8
// CHECK-SAME: (i8{{.+}}%opta, i8{{.+}}%optb)
#[no_mangle]
pub fn option_or_nz_u8(
    opta: Option<NonZero<u8>>,
    optb: Option<NonZero<u8>>,
) -> Option<NonZero<u8>> {
    // CHECK: start:
    // CHECK-NEXT: [[NOT_A:%.+]] = icmp eq i8 %opta, 0
    // CHECK-NEXT: [[R:%.+]] = select i1 [[NOT_A]], i8 %optb, i8 %opta
    // CHECK: ret i8 [[R]]
    opta.or(optb)
}

// CHECK-LABEL: @if_some_nz_u8
// CHECK-SAME: (i8{{.+}}%opta, i8{{.+}}%optb)
#[no_mangle]
pub fn if_some_nz_u8(opta: Option<NonZero<u8>>, optb: Option<NonZero<u8>>) -> Option<NonZero<u8>> {
    // CHECK: start:
    // CHECK-NEXT: [[NOT_A:%.+]] = icmp eq i8 %opta, 0
    // CHECK-NEXT: [[R:%.+]] = select i1 [[NOT_A]], i8 %optb, i8 %opta
    // CHECK: ret i8 [[R]]
    if opta.is_some() { opta } else { optb }
}
