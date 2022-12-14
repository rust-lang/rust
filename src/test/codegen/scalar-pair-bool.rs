// compile-flags: -O

#![crate_type = "lib"]

// CHECK: define{{.*}}{ i8, i8 } @pair_bool_bool(i1 noundef zeroext %pair.0, i1 noundef zeroext %pair.1)
#[no_mangle]
pub fn pair_bool_bool(pair: (bool, bool)) -> (bool, bool) {
    pair
}

// CHECK: define{{.*}}{ i8, i32 } @pair_bool_i32(i1 noundef zeroext %pair.0, i32 %pair.1)
#[no_mangle]
pub fn pair_bool_i32(pair: (bool, i32)) -> (bool, i32) {
    pair
}

// CHECK: define{{.*}}{ i32, i8 } @pair_i32_bool(i32 %pair.0, i1 noundef zeroext %pair.1)
#[no_mangle]
pub fn pair_i32_bool(pair: (i32, bool)) -> (i32, bool) {
    pair
}

// CHECK: define{{.*}}{ i8, i8 } @pair_and_or(i1 noundef zeroext %_1.0, i1 noundef zeroext %_1.1)
#[no_mangle]
pub fn pair_and_or((a, b): (bool, bool)) -> (bool, bool) {
    // Make sure it can operate directly on the unpacked args
    // (but it might not be using simple and/or instructions)
    // CHECK-DAG: %_1.0
    // CHECK-DAG: %_1.1
    (a && b, a || b)
}

// CHECK: define{{.*}}void @pair_branches(i1 noundef zeroext %_1.0, i1 noundef zeroext %_1.1)
#[no_mangle]
pub fn pair_branches((a, b): (bool, bool)) {
    // Make sure it can branch directly on the unpacked bool args
    // CHECK: br i1 %_1.0
    if a {
        println!("Hello!");
    }
    // CHECK: br i1 %_1.1
    if b {
        println!("Goodbye!");
    }
}
