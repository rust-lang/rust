#![no_std]
#![no_main]
//@compile-flags: -Cpanic=abort
//@ignore-target: windows # no-std not supported on Windows

#[path = "../../utils/mod.no_std.rs"]
mod utils;

extern "Rust" fn thread_start(_null: *mut ()) {
    unsafe {
        utils::miri_spin_loop();
        utils::miri_spin_loop();
    }
}

#[no_mangle]
fn miri_start(_argc: isize, _argv: *const *const u8) -> isize {
    unsafe {
        let thread_id = utils::miri_thread_spawn(thread_start, core::ptr::null_mut());
        assert_eq!(utils::miri_thread_join(thread_id), true);
    }
    0
}

#[panic_handler]
fn panic_handler(_: &core::panic::PanicInfo) -> ! {
    loop {}
}
