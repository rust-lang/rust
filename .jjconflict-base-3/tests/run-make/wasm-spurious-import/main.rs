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
