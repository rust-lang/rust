// Tests output of multiple permutations of `Option::or`
//@ compile-flags: -Copt-level=3 -Zmerge-functions=disabled

#![crate_type = "lib"]

// CHECK-LABEL: @or_match_u8
#[no_mangle]
pub fn or_match_u8(opta: Option<u8>, optb: Option<u8>) -> Option<u8> {
    // CHECK: start:
    // CHECK-NEXT: or i1 %0
    // CHECK-NEXT: select i1 %0
    // CHECK-NEXT: insertvalue { i1, i8 }
    // CHECK-NEXT: insertvalue { i1, i8 }
    // ret { i1, i8 }
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
    // CHECK-NEXT: select i1
    // CHECK-NEXT: or i1
    // CHECK-NEXT: insertvalue { i1, i8 }
    // CHECK-NEXT: insertvalue { i1, i8 }
    // ret { i1, i8 }
    match opta {
        Some(_) => opta,
        None => optb,
    }
}

// CHECK-LABEL: @option_or_u8
#[no_mangle]
pub fn option_or_u8(opta: Option<u8>, optb: Option<u8>) -> Option<u8> {
    // CHECK: start:
    // CHECK-NEXT: select i1
    // CHECK-NEXT: or i1
    // CHECK-NEXT: insertvalue { i1, i8 }
    // CHECK-NEXT: insertvalue { i1, i8 }
    // ret { i1, i8 }
    opta.or(optb)
}

// CHECK-LABEL: @if_some_u8
#[no_mangle]
pub fn if_some_u8(opta: Option<u8>, optb: Option<u8>) -> Option<u8> {
    // CHECK: start:
    // CHECK-NEXT: select i1
    // CHECK-NEXT: or i1
    // CHECK-NEXT: insertvalue { i1, i8 }
    // CHECK-NEXT: insertvalue { i1, i8 }
    // ret { i1, i8 }
    if opta.is_some() { opta } else { optb }
}

// CHECK-LABEL: @or_match_slice_u8
// CHECK-SAME: (i16 %0, i16 %1)
#[no_mangle]
pub fn or_match_slice_u8(opta: Option<[u8; 1]>, optb: Option<[u8; 1]>) -> Option<[u8; 1]> {
    // CHECK: start:
    // CHECK-NEXT: trunc i16 %0 to i1
    // CHECK-NEXT: select i1 %2, i16 %0, i16 %1
    // ret i16
    match opta {
        Some(x) => Some(x),
        None => optb,
    }
}

// CHECK-LABEL: @or_match_slice_alt_u8
#[no_mangle]
pub fn or_match_slice_alt_u8(opta: Option<[u8; 1]>, optb: Option<[u8; 1]>) -> Option<[u8; 1]> {
    // CHECK: start:
    // CHECK-NEXT: trunc i16 %0 to i1
    // CHECK-NEXT: select i1 %2, i16 %0, i16 %1
    // ret i16
    match opta {
        Some(_) => opta,
        None => optb,
    }
}

// CHECK-LABEL: @option_or_slice_u8
#[no_mangle]
pub fn option_or_slice_u8(opta: Option<[u8; 1]>, optb: Option<[u8; 1]>) -> Option<[u8; 1]> {
    // CHECK: start:
    // CHECK-NEXT: trunc i16 %0 to i1
    // CHECK-NEXT: select i1 %2, i16 %0, i16 %1
    // ret i16
    opta.or(optb)
}

// CHECK-LABEL: @if_some_slice_u8
#[no_mangle]
pub fn if_some_slice_u8(opta: Option<[u8; 1]>, optb: Option<[u8; 1]>) -> Option<[u8; 1]> {
    // CHECK: start:
    // CHECK-NEXT: trunc i16 %0 to i1
    // CHECK-NEXT: select i1 %2, i16 %0, i16 %1
    // ret i16
    if opta.is_some() { opta } else { optb }
}

pub struct Test {
    _a: u8,
    _b: u8,
}

// CHECK-LABEL: @or_match_type
#[no_mangle]
pub fn or_match_type(opta: Option<Test>, optb: Option<Test>) -> Option<Test> {
    // CHECK: start:
    // CHECK-NEXT: trunc i24 %0 to i1
    // CHECK-NEXT: select i1 %2, i24 %0, i24 %1
    // ret i24
    match opta {
        Some(x) => Some(x),
        None => optb,
    }
}

// CHECK-LABEL: @or_match_alt_type
#[no_mangle]
pub fn or_match_alt_type(opta: Option<Test>, optb: Option<Test>) -> Option<Test> {
    // CHECK: start:
    // CHECK-NEXT: trunc i24 %0 to i1
    // CHECK-NEXT: select i1 %2, i24 %0, i24 %1
    // ret i24
    match opta {
        Some(_) => opta,
        None => optb,
    }
}

// CHECK-LABEL: @option_or_type
#[no_mangle]
pub fn option_or_type(opta: Option<Test>, optb: Option<Test>) -> Option<Test> {
    // CHECK: start:
    // CHECK-NEXT: trunc i24 %0 to i1
    // CHECK-NEXT: select i1 %2, i24 %0, i24 %1
    // ret i24
    opta.or(optb)
}

// CHECK-LABEL: @if_some_type
#[no_mangle]
pub fn if_some_type(opta: Option<Test>, optb: Option<Test>) -> Option<Test> {
    // CHECK: start:
    // CHECK-NEXT: trunc i24 %0 to i1
    // CHECK-NEXT: select i1 %2, i24 %0, i24 %1
    // ret i24
    if opta.is_some() { opta } else { optb }
}
