//@ compile-flags: -Zmir-opt-level=0 -C no-prepopulate-passes -Copt-level=0
// make sure that branching on a constant does not emit a conditional
// branch or a switch

#![crate_type = "lib"]

// CHECK-LABEL: @if_bool
#[no_mangle]
pub fn if_bool() {
    // CHECK-NOT: br i1
    // CHECK-NOT: switch
    _ = if true { 0 } else { 1 };

    _ = if false { 0 } else { 1 };
}

// CHECK-LABEL: @if_constant_int_eq
#[no_mangle]
pub fn if_constant_int_eq() {
    // CHECK-NOT: br i1
    // CHECK-NOT: switch
    let val = 0;
    _ = if val == 0 { 0 } else { 1 };

    // CHECK: br label %{{.+}}
    _ = if val == 1 { 0 } else { 1 };
}

// CHECK-LABEL: @if_constant_match
#[no_mangle]
pub fn if_constant_match() {
    // CHECK-NOT: br i1
    // CHECK-NOT: switch
    _ = match 1 {
        1 => 2,
        2 => 3,
        _ => 4,
    };

    _ = match 1 {
        2 => 3,
        _ => 4,
    };

    _ = match -1 {
        -1 => 1,
        _ => 0,
    }
}
