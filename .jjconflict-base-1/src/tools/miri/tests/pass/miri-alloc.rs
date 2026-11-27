#![no_std]
#![no_main]
//@compile-flags: -Cpanic=abort
// windows tls dtors go through libstd right now, thus this test
// cannot pass. When windows tls dtors go through the special magic
// windows linker section, we can run this test on windows again.
//@ignore-target: windows # no-std not supported on Windows

#[path = "../utils/mod.no_std.rs"]
mod utils;

#[no_mangle]
fn miri_start(_argc: isize, _argv: *const *const u8) -> isize {
    unsafe {
        let ptr = utils::miri_alloc(123, 1);
        core::ptr::write_bytes(ptr, 0u8, 123);
        utils::miri_dealloc(ptr, 123, 1);
    }
    0
}

#[panic_handler]
fn panic_handler(_: &core::panic::PanicInfo) -> ! {
    loop {}
}
