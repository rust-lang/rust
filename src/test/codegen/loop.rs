// compile-flags: -C opt-level=3

#![crate_type = "lib"]

// CHECK-LABEL: @check_loop
#[no_mangle]
pub fn check_loop() -> u8 {
    // CHECK-NOT: unreachable
    call_looper()
}

#[no_mangle]
fn call_looper() -> ! {
    loop {}
}
