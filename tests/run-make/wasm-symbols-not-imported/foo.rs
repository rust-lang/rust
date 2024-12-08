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
