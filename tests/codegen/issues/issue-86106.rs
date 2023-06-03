// min-llvm-version: 15.0
// only-64bit llvm appears to use stores instead of memset on 32bit
// compile-flags: -C opt-level=3 -Z merge-functions=disabled

// The below two functions ensure that both `String::new()` and `"".to_string()`
// produce the identical code.

#![crate_type = "lib"]

// CHECK-LABEL: define {{(dso_local )?}}void @string_new
#[no_mangle]
pub fn string_new() -> String {
    // CHECK: store ptr inttoptr
    // CHECK-NEXT: getelementptr
    // CHECK-NEXT: call void @llvm.memset
    // CHECK-NEXT: ret void
    String::new()
}

// CHECK-LABEL: define {{(dso_local )?}}void @empty_to_string
#[no_mangle]
pub fn empty_to_string() -> String {
    // CHECK: store ptr inttoptr
    // CHECK-NEXT: getelementptr
    // CHECK-NEXT: call void @llvm.memset
    // CHECK-NEXT: ret void
    "".to_string()
}

// The below two functions ensure that both `vec![]` and `vec![].clone()`
// produce the identical code.

// CHECK-LABEL: @empty_vec
#[no_mangle]
pub fn empty_vec() -> Vec<u8> {
    // CHECK: store ptr inttoptr
    // CHECK-NEXT: getelementptr
    // CHECK-NEXT: call void @llvm.memset
    // CHECK-NEXT: ret void
    vec![]
}

// CHECK-LABEL: @empty_vec_clone
#[no_mangle]
pub fn empty_vec_clone() -> Vec<u8> {
    // CHECK: store ptr inttoptr
    // CHECK-NEXT: getelementptr
    // CHECK-NEXT: call void @llvm.memset
    // CHECK-NEXT: ret void
    vec![].clone()
}
