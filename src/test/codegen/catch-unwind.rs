// compile-flags: -O

#![crate_type = "lib"]

extern "C" {
    fn bar();
}

// CHECK-LABEL: @foo
#[no_mangle]
pub unsafe fn foo() -> i32 {
    // CHECK: call void @bar
    // CHECK: ret i32 0
    std::panic::catch_unwind(|| {
        bar();
        0
    })
    .unwrap()
}
