//@ compile-flags: -C opt-level=3 -Z merge-functions=disabled
#![crate_type = "lib"]

//! Make sure str::to_string is specialized not to use fmt machinery.

// CHECK-LABEL: define {{(dso_local )?}}void @one_ref
#[no_mangle]
pub fn one_ref(input: &str) -> String {
    // CHECK-NOT: {{(call|invoke).*}}fmt
    input.to_string()
}

// CHECK-LABEL: define {{(dso_local )?}}void @two_ref
#[no_mangle]
pub fn two_ref(input: &&str) -> String {
    // CHECK-NOT: {{(call|invoke).*}}fmt
    input.to_string()
}

// CHECK-LABEL: define {{(dso_local )?}}void @thirteen_ref
#[no_mangle]
pub fn thirteen_ref(input: &&&&&&&&&&&&&str) -> String {
    // CHECK-NOT: {{(call|invoke).*}}fmt
    input.to_string()
}

// This is a known performance cliff because of the macro-generated
// specialized impl. If this test suddenly starts failing,
// consider removing the `to_string_str!` macro in `alloc/str/string.rs`.
//
// CHECK-LABEL: define {{(dso_local )?}}void @fourteen_ref
#[no_mangle]
pub fn fourteen_ref(input: &&&&&&&&&&&&&&str) -> String {
    // CHECK: {{(call|invoke).*}}fmt
    input.to_string()
}
