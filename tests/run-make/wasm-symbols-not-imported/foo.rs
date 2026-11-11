#![crate_type = "cdylib"]
#![no_std]

use core::panic::PanicInfo;

#[no_mangle]
pub extern "C" fn foo() {
    panic!()
}

#[panic_handler]
fn panic(_info: &PanicInfo) -> ! {
    loop {}
}

#[cfg(target_env = "p3")]
#[no_mangle]
pub extern "C" fn __wasm_task_hook(_: u32) {}
#[cfg(target_env = "p3")]
#[no_mangle]
pub extern "C" fn cabi_realloc(_: *mut u8, _: usize, _: usize, _: usize) -> *mut u8 {
    loop {}
}
