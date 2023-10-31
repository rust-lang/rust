// only-wasm32-bare
// compile-flags: -C panic=unwind

#![crate_type = "lib"]
#![feature(core_intrinsics)]
#![feature(rustc_attrs)]

extern {
    fn may_panic();

    #[rustc_nounwind]
    fn log_number(number: usize);
}

struct LogOnDrop;

impl Drop for LogOnDrop {
    fn drop(&mut self) {
        unsafe { log_number(0); }
    }
}

// CHECK-LABEL: @test_cleanup() {{.*}} @__gxx_wasm_personality_v0
#[no_mangle]
pub fn test_cleanup() {
    let _log_on_drop = LogOnDrop;
    unsafe { may_panic(); }

    // CHECK-NOT: call
    // CHECK: invoke void @may_panic()
    // CHECK: %cleanuppad = cleanuppad within none []
}

// CHECK-LABEL: @test_rtry() {{.*}} @__gxx_wasm_personality_v0
#[no_mangle]
pub fn test_rtry() {
    unsafe {
        core::intrinsics::r#try(|_| {
            may_panic();
        }, core::ptr::null_mut(), |data, exception| {
            log_number(data as usize);
            log_number(exception as usize);
        });
    }

    // CHECK-NOT: call
    // CHECK: invoke void @may_panic()
    // CHECK: {{.*}} = catchswitch within none [label {{.*}}] unwind to caller
    // CHECK: {{.*}} = catchpad within {{.*}} [ptr null]
    // CHECK: catchret
}
