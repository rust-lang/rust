#![allow(unsafe_op_in_unsafe_fn)]

use crate::io;

pub(crate) fn map_motor_error(err: moto_rt::Error) -> io::Error {
    let error_code: moto_rt::ErrorCode = err.into();
    io::Error::from_raw_os_error(error_code.into())
}

// Weak: when a program is linked with mlibc (e.g. via the Motor clang
// driver, which always links mlibc's crt1.o), crt1.o's strong motor_start
// must win. mlibc's entry initializes the VDSO vtable and the C runtime
// (TCB, stdio, .init_array constructors) and then calls the C `main`
// that rustc generates, so Rust std works identically in both flows.
#[cfg(not(test))]
#[unsafe(no_mangle)]
#[linkage = "weak"]
pub extern "C" fn motor_start() -> ! {
    // Initialize the runtime.
    moto_rt::start();

    // Call main.
    unsafe extern "C" {
        fn main(_: isize, _: *const *const u8, _: u8) -> i32;
    }
    let result = unsafe { main(0, core::ptr::null(), 0) };

    // Terminate the process.
    moto_rt::process::exit(result)
}

// SAFETY: must be called only once during runtime initialization.
// NOTE: Motor OS uses moto_rt::start() to initialize runtime (see above).
pub unsafe fn init(_argc: isize, _argv: *const *const u8, _sigpipe: u8) {}

// SAFETY: must be called only once during runtime cleanup.
// NOTE: this is not guaranteed to run, for example when the program aborts.
pub unsafe fn cleanup() {}

pub fn unsupported<T>() -> io::Result<T> {
    Err(unsupported_err())
}

pub fn unsupported_err() -> io::Error {
    io::Error::UNSUPPORTED_PLATFORM
}

pub fn abort_internal() -> ! {
    moto_rt::process::exit(-1)
}
