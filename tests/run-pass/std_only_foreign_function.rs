//! Make sure we can call foreign functions that are only allowed within libstd if we are "libstd"
//! (defining the `start` lang item).
#![feature(lang_items, rustc_private, core_intrinsics)]
#![no_std]

use core::{intrinsics, panic::PanicInfo};

#[lang = "eh_personality"]
fn rust_eh_personality() {}

#[panic_handler]
fn panic_handler(_: &PanicInfo<'_>) -> ! {
    intrinsics::abort()
}

#[lang = "start"]
fn start(main: fn(), _argc: isize, _argv: *const *const u8) -> isize {
    main();
    0
}

fn main() {
    #[cfg(unix)]
    unsafe {
        extern crate libc;
        assert_eq!(libc::signal(libc::SIGPIPE, libc::SIG_IGN), 0);
    }
    #[cfg(windows)]
    unsafe {
        extern "system" {
            fn GetProcessHeap() -> *mut core::ffi::c_void;
            fn ExitProcess(code: u32) -> !;
        }
        assert_eq!(GetProcessHeap() as usize, 1);
        // Early exit to avoid the requirement of
        // `std::sys::windows::thread_local_key::p_thread_callback`.
        ExitProcess(0);
    }
}
