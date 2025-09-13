//@ only-wasm32
//@ revisions: WASM WASMEXN
//@ [WASMEXN] compile-flags: -C panic=unwind -Z emscripten-wasm-eh

#![crate_type = "lib"]
#![feature(core_intrinsics, wasm_exception_handling_intrinsics, link_llvm_intrinsics)]

extern "C-unwind" {
    fn may_panic();
}

extern "C" {
    fn log_number(number: usize);
}

struct LogOnDrop;

impl Drop for LogOnDrop {
    fn drop(&mut self) {
        unsafe {
            log_number(0);
        }
    }
}

// WASM-LABEL: @test_cleanup() {{.*}}
// WASMEXN-LABEL: @test_cleanup() {{.*}} @__gxx_wasm_personality_v0
#[no_mangle]
pub fn test_cleanup() {
    let _log_on_drop = LogOnDrop;
    unsafe {
        may_panic();
    }

    // WASMEXN-NOT: call
    // WASMEXN: invoke void @may_panic()
    // WASMEXN: %cleanuppad = cleanuppad within none []
    //
    // WASM: call void @may_panic()
    // WASM-NOT: invoke void @may_panic()
}

// WASM-LABEL: @test_rtry() {{.*}}
// WASMEXN-LABEL: @test_rtry() {{.*}} @__gxx_wasm_personality_v0
#[no_mangle]
pub fn test_rtry() {
    unsafe {
        core::intrinsics::catch_unwind(
            |_| {
                may_panic();
            },
            core::ptr::null_mut(),
            |data, exception| {
                log_number(data as usize);
                log_number(exception as usize);
            },
        );
    }

    // WASMEXN-NOT: call
    // WASMEXN: invoke void @may_panic()
    // WASMEXN: {{.*}} = catchswitch within none [label {{.*}}] unwind to caller
    // WASMEXN: {{.*}} = catchpad within {{.*}} [ptr null]
    // WASMEXN: catchret

    // WASM: call void @may_panic()
    // WASM-NOT: invoke void @may_panic()
    // WASM-NOT: catchswitch
    // WASM-NOT: catchpad
    // WASM-NOT: catchret
}

// Make sure the intrinsic is not inferred as nounwind. This is a regression test for #132416.
//
// Note that this test uses the raw `wasm_throw` intrinsic because the one from
// libstd was built with `-Cpanic=abort` and it's technically not valid to use
// when this crate is compiled with `-Cpanic=unwind`.
//
// WASMEXN-LABEL: @test_intrinsic() {{.*}} @__gxx_wasm_personality_v0
#[no_mangle]
#[cfg(wasmexn)]
pub fn test_intrinsic() {
    let _log_on_drop = LogOnDrop;

    unsafe extern "C-unwind" {
        #[link_name = "llvm.wasm.throw"]
        fn wasm_throw(tag: i32, ptr: *mut u8) -> !;
    }
    unsafe {
        wasm_throw(0, core::ptr::null_mut());
    }

    // WASMEXN-NOT: call
    // WASMEXN: invoke void @llvm.wasm.throw(i32 noundef 0, ptr noundef null)
    // WASMEXN: %cleanuppad = cleanuppad within none []
}
