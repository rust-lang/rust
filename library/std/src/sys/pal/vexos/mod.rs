#[path = "../unsupported/os.rs"]
pub mod os;
#[path = "../unsupported/pipe.rs"]
pub mod pipe;
pub mod thread;
pub mod time;

use crate::arch::global_asm;
use crate::ptr::{self, addr_of_mut};
use crate::time::{Duration, Instant};

global_asm!(
    r#"
    .section .boot, "ax"
    .global _boot

    _boot:
        ldr sp, =__stack_top @ Set up the user stack.
        b _start             @ Jump to the Rust entrypoint.
    "#
);

#[cfg(not(test))]
#[no_mangle]
pub unsafe extern "C" fn _start() -> ! {
    extern "C" {
        static mut __bss_start: u8;
        static mut __bss_end: u8;

        fn main() -> i32;
    }

    // Clear the .bss (uninitialized statics) section by filling it with zeroes.
    // This is required, since the compiler assumes it will be zeroed on first access.
    ptr::write_bytes(
        &raw mut __bss_start,
        0,
        (&raw mut __bss_end).offset_from(&raw mut __bss_start) as usize,
    );

    main();

    abort_internal()
}

// SAFETY: must be called only once during runtime initialization.
// NOTE: this is not guaranteed to run, for example when Rust code is called externally.
pub unsafe fn init(_argc: isize, _argv: *const *const u8, _sigpipe: u8) {}

// SAFETY: must be called only once during runtime cleanup.
// NOTE: this is not guaranteed to run, for example when the program aborts.
pub unsafe fn cleanup() {}

pub fn unsupported<T>() -> crate::io::Result<T> {
    Err(unsupported_err())
}

pub fn unsupported_err() -> crate::io::Error {
    crate::io::Error::UNSUPPORTED_PLATFORM
}

pub fn is_interrupted(_code: i32) -> bool {
    false
}

pub fn decode_error_kind(_code: i32) -> crate::io::ErrorKind {
    crate::io::ErrorKind::Uncategorized
}

pub fn abort_internal() -> ! {
    let exit_time = Instant::now();
    const FLUSH_TIMEOUT: Duration = Duration::from_millis(15);

    unsafe {
        // Force the serial buffer to flush
        while exit_time.elapsed() < FLUSH_TIMEOUT {
            vex_sdk::vexTasksRun();

            // If the buffer has been fully flushed, exit the loop
            if vex_sdk::vexSerialWriteFree(stdio::STDIO_CHANNEL) == (stdio::STDOUT_BUF_SIZE as i32)
            {
                break;
            }
        }

        vex_sdk::vexSystemExitRequest();

        loop {
            vex_sdk::vexTasksRun();
        }
    }
}
