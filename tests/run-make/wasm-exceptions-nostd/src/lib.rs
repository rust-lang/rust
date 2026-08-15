#![no_std]
#![crate_type = "cdylib"]
// Allow a few unstable features because we create a panic
// runtime for native wasm exceptions from scratch
#![feature(core_intrinsics)]
#![feature(lang_items)]
#![feature(link_llvm_intrinsics)]

extern crate alloc;

/// This module allows us to use `Box`, `String`, ... even in no-std
mod arena_alloc;

/// This module allows logging text, even in no-std
mod logging;

/// This module allows exceptions, even in no-std
#[cfg(target_arch = "wasm32")]
mod panicking;

use alloc::boxed::Box;
use alloc::string::String;

struct LogOnDrop;

impl Drop for LogOnDrop {
    fn drop(&mut self) {
        logging::log_str("Dropped");
    }
}

#[allow(unreachable_code)]
#[allow(unconditional_panic)]
#[no_mangle]
pub extern "C" fn start() -> usize {
    let data = 0x1234usize as *mut u8; // Something to recognize

    unsafe {
        core::intrinsics::catch_unwind(
            |data: *mut u8| {
                let _log_on_drop = LogOnDrop;

                logging::log_str(&alloc::format!("`r#try` called with ptr {:?}", data));
                let x = [12];
                let _ = x[4]; // should panic

                logging::log_str("This line should not be visible! :(");
            },
            data,
            |data, exception| {
                let exception = *Box::from_raw(exception as *mut String);
                logging::log_str("Caught something!");
                logging::log_str(&alloc::format!("  data     : {:?}", data));
                logging::log_str(&alloc::format!("  exception: {:?}", exception));
            },
        );
    }

    logging::log_str("This program terminates correctly.");
    0
}
