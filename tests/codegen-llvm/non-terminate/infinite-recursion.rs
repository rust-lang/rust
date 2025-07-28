//@ compile-flags: -C opt-level=3

#![crate_type = "lib"]
#![allow(unconditional_recursion)]

// CHECK-LABEL: @infinite_recursion
#[no_mangle]
fn infinite_recursion() -> u8 {
    // CHECK-NOT: ret i8 undef
    // CHECK: br label %{{.+}}
    // CHECK-NOT: ret i8 undef
    infinite_recursion()
}
