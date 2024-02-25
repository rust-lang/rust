//@ compile-flags: -Zmir-opt-level=0 -C no-prepopulate-passes -Copt-level=0
// make sure that branching on a constant does not emit a conditional
// branch or a switch

#![crate_type = "lib"]

// CHECK-LABEL: @if_bool
#[no_mangle]
pub fn if_bool() {
    // CHECK: br label %{{.+}}
    _ = if true {
        0
    } else {
        1
    };

    // CHECK: br label %{{.+}}
    _ = if false {
        0
    } else {
        1
    };
}

// CHECK-LABEL: @if_constant_int_eq
#[no_mangle]
pub fn if_constant_int_eq() {
    let val = 0;
    // CHECK: br label %{{.+}}
    _ = if val == 0 {
        0
    } else {
        1
    };

    // CHECK: br label %{{.+}}
    _ = if val == 1 {
        0
    } else {
        1
    };
}

// CHECK-LABEL: @if_constant_match
#[no_mangle]
pub fn if_constant_match() {
    // CHECK: br label %{{.+}}
    _ = match 1 {
        1 => 2,
        2 => 3,
        _ => 4
    };

    // CHECK: br label %{{.+}}
    _ = match 1 {
        2 => 3,
        _ => 4
    };

    // CHECK: br label %[[MINUS1:.+]]
    _ = match -1 {
    // CHECK: [[MINUS1]]:
    // CHECK: store i32 1
        -1 => 1,
        _ => 0,
    }
}
