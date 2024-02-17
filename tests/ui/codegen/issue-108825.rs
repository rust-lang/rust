// Test that linking a no_std application still outputs the
// `native-static-libs: ` note, even though it's empty.
//@ compile-flags: -Cpanic=abort --print=native-static-libs
//@ build-pass
#![crate_type = "staticlib"]
#![no_std]

#[panic_handler]
fn panic(_info: &core::panic::PanicInfo) -> ! {
    loop {}
}
