//@ only-wasm32
//@ assembly-output: emit-asm
//@ compile-flags: -C target-feature=+exception-handling
//@ compile-flags: -C panic=unwind

#![crate_type = "lib"]
#![feature(core_intrinsics)]

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

// CHECK-LABEL: test_cleanup:
#[no_mangle]
pub fn test_cleanup() {
    let _log_on_drop = LogOnDrop;
    unsafe {
        may_panic();
    }

    // CHECK-NOT: call
    // CHECK: try_table (catch_all_ref 0)
    // CHECK: call may_panic
    // CHECK: end_try_table
    // CHECK: call log_number
    // CHECK: throw_ref
}

// CHECK-LABEL: test_rtry:
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

    // CHECK-NOT: call
    // CHECK: try_table (catch __cpp_exception 0)
    // CHECK: call may_panic
    // CHECK: end_try_table
    // CHECK: call log_number
    // CHECK: call log_number
    // CHECK-NOT: throw_ref
    // CHECK: end_function
}
