//@ compile-flags: -C opt-level=3
//@ min-llvm-version: 21

#![crate_type = "lib"]
#![allow(unused_assignments, unused_variables)]

// Check that the old string is deallocated, but a new one is not initialized.
#[unsafe(no_mangle)]
pub fn test_str_new(mut s: String) {
    // CHECK-LABEL: @test_str_new
    // CHECK: __rust_dealloc
    // CHECK-NOT: store
    s = String::new();
}

#[unsafe(no_mangle)]
pub fn test_str_take(mut x: String) -> String {
    // CHECK-LABEL: @test_str_take
    // CHECK-NEXT: {{.*}}:
    // CHECK-NEXT: call void @llvm.memcpy
    // CHECK-NEXT: ret
    core::mem::take(&mut x)
}

#[unsafe(no_mangle)]
pub fn test_array_store(mut x: [u32; 100]) {
    // CHECK-LABEL: @test_array_store
    // CHECK-NEXT: {{.*}}:
    // CHECK-NEXT: ret
    x[0] = 1;
}
