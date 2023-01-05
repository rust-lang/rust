// min-llvm-version: 15.0
// compile-flags: -C opt-level=3 -Z merge-functions=disabled

// The below two functions ensure that both `String::new()` and `"".to_string()`
// produce the identical code.

#![crate_type = "lib"]

// CHECK-LABEL: define void @string_new
#[no_mangle]
pub fn string_new() -> String {
    // CHECK-NOT: load i8
    // CHECK: store i{{32|64}}
    // CHECK-NEXT: getelementptr
    // CHECK-NEXT: store ptr
    // CHECK-NEXT: getelementptr
    // CHECK-NEXT: store i{{32|64}}
    // CHECK-NEXT: ret void
    String::new()
}

// CHECK-LABEL: define void @empty_to_string
#[no_mangle]
pub fn empty_to_string() -> String {
    // CHECK-NOT: load i8
    // CHECK: store i{{32|64}}
    // CHECK-NEXT: getelementptr
    // CHECK-NEXT: store ptr
    // CHECK-NEXT: getelementptr
    // CHECK-NEXT: store i{{32|64}}
    // CHECK-NEXT: ret void
    "".to_string()
}

// The below two functions ensure that both `vec![]` and `vec![].clone()`
// produce the identical code.

// CHECK-LABEL: @empty_vec
#[no_mangle]
pub fn empty_vec() -> Vec<u8> {
    // CHECK: store i{{32|64}}
    // CHECK-NOT: load i8
    // CHECK-NEXT: getelementptr
    // CHECK-NEXT: store ptr
    // CHECK-NEXT: getelementptr
    // CHECK-NEXT: store i{{32|64}}
    // CHECK-NEXT: ret void
    vec![]
}

// CHECK-LABEL: @empty_vec_clone
#[no_mangle]
pub fn empty_vec_clone() -> Vec<u8> {
    // CHECK: store i{{32|64}}
    // CHECK-NOT: load i8
    // CHECK-NEXT: getelementptr
    // CHECK-NEXT: store ptr
    // CHECK-NEXT: getelementptr
    // CHECK-NEXT: store i{{32|64}}
    // CHECK-NEXT: ret void
    vec![].clone()
}
