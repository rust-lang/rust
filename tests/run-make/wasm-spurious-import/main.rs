#![crate_type = "cdylib"]
#![no_std]

#[panic_handler]
fn my_panic(_info: &core::panic::PanicInfo) -> ! {
    loop {}
}

#[no_mangle]
pub fn multer(a: i128, b: i128) -> i128 {
    // Trigger usage of the __multi3 compiler intrinsic which then leads to an imported
    // panic function in case of a bug. We verify that no imports exist in our verifier.
    a * b
}

#[cfg(target_env = "p3")]
#[no_mangle]
pub extern "C" fn __wasm_task_hook(_: u32) {}
#[cfg(target_env = "p3")]
#[no_mangle]
pub extern "C" fn cabi_realloc(_: *mut u8, _: usize, _: usize, _: usize) -> *mut u8 {
    loop {}
}
