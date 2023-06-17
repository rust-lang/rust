// only-64bit llvm appears to use stores instead of memset on 32bit
// compile-flags: -C opt-level=3 -Z merge-functions=disabled

// The below two functions ensure that both `String::new()` and `"".to_string()`
// generate their values directly, rather that creating a constant and copying
// that constant (which takes more instructions because of PIC).

#![crate_type = "lib"]

// CHECK-LABEL: define {{(dso_local )?}}void @string_new
#[no_mangle]
pub fn string_new() -> String {
    // CHECK: store {{i16|i32|i64}} 1, ptr %_0,
    // CHECK-NEXT: getelementptr
    // CHECK-NEXT: call void @llvm.memset
    // CHECK-NEXT: ret void
    String::new()
}

// CHECK-LABEL: define {{(dso_local )?}}void @empty_to_string
#[no_mangle]
pub fn empty_to_string() -> String {
    // CHECK: store {{i48|i96|i192}} 1, ptr %_0, align {{2|4|8}}
    // CHECK-NEXT: ret
    "".to_string()
}

// The below two functions ensure that both `vec![]` and `vec![].clone()`
// produce the identical code.

// CHECK-LABEL: @empty_vec
#[no_mangle]
pub fn empty_vec() -> Vec<u8> {
    // CHECK: store ptr inttoptr ({{i16|i32|i64}} 1 to ptr), ptr %_0,
    // CHECK-NEXT: getelementptr
    // CHECK-NEXT: call void @llvm.memset
    // CHECK-NEXT: ret void
    vec![]
}

// CHECK-LABEL: @empty_vec_clone
#[no_mangle]
pub fn empty_vec_clone() -> Vec<u8> {
    // CHECK: store {{i16|i32|i64}} 1, ptr %_0,
    // CHECK-NEXT: getelementptr
    // CHECK-NEXT: call void @llvm.memset
    // CHECK-NEXT: ret void
    vec![].clone()
}

// CHECK-LABEL: @empty_vec_from_array
#[no_mangle]
pub fn empty_vec_from_array() -> Vec<u8> {
    // CHECK: store ptr inttoptr ({{i16|i32|i64}} 1 to ptr), ptr %_0,
    // CHECK-NEXT: getelementptr
    // CHECK-NEXT: call void @llvm.memset
    // CHECK-NEXT: ret void
    [].into()
}
