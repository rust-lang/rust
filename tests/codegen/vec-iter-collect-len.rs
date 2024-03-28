//@ compile-flags: -O
#![crate_type="lib"]

// CHECK-NOT: call

// CHECK-LABEL: @collect_calls_alloc
#[no_mangle]
pub fn collect_calls_alloc() -> Vec<&'static i32> {
    // CHECK: call{{.+}}@__rust_alloc(
    [1, 2, 3].iter().collect()
}

// CHECK-LABEL: @collect_then_len_avoids_alloc
#[no_mangle]
pub fn collect_then_len_avoids_alloc() -> usize {
    // CHECK-NOT: call{{.+}}@__rust_alloc(
    let v: Vec<&'static i32> = [1, 2, 3].iter().collect();
    v.len()
}
