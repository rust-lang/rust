//@ compile-flags: -Copt-level=3
#![crate_type = "lib"]

#[no_mangle]
pub fn should_use_low(a: [i32; 10], b: [i32; 100], p: fn(i32) -> bool) -> Vec<i32> {
    // CHECK-LABEL: define void @should_use_low
    // CHECK: call{{.+}}dereferenceable_or_null(40){{.+}}@__rust_alloc(
    a.iter().copied().chain(b.iter().copied().filter(|x| p(*x))).collect()
}

#[no_mangle]
pub fn should_use_high(a: [i32; 100], b: [i32; 10], p: fn(i32) -> bool) -> Vec<i32> {
    // CHECK-LABEL: define void @should_use_high
    // CHECK: call{{.+}}dereferenceable_or_null(440){{.+}}@__rust_alloc(
    a.iter().copied().chain(b.iter().copied().filter(|x| p(*x))).collect()
}
