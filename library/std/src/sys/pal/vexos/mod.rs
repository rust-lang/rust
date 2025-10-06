#[path = "../unsupported/os.rs"]
pub mod os;
#[path = "../unsupported/pipe.rs"]
pub mod pipe;
pub mod time;

#[expect(dead_code)]
#[path = "../unsupported/common.rs"]
mod unsupported_common;

pub use unsupported_common::{
    decode_error_kind, init, is_interrupted, unsupported, unsupported_err,
};

use crate::arch::global_asm;
use crate::ptr;
use crate::sys::stdio;
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
#[unsafe(no_mangle)]
pub unsafe extern "C" fn _start() -> ! {
    unsafe extern "C" {
        static mut __bss_start: u8;
        static mut __bss_end: u8;

        fn main() -> i32;
    }

    // Clear the .bss (uninitialized statics) section by filling it with zeroes.
    // This is required, since the compiler assumes it will be zeroed on first access.
    ptr::write_bytes(
        &raw mut __bss_start,
        0,
        (&raw mut __bss_end).offset_from_unsigned(&raw mut __bss_start),
    );

    main();

    cleanup();
    abort_internal()
}

// SAFETY: must be called only once during runtime cleanup.
// NOTE: this is not guaranteed to run, for example when the program aborts.
pub unsafe fn cleanup() {
    let exit_time = Instant::now();
    const FLUSH_TIMEOUT: Duration = Duration::from_millis(15);

    // Force the serial buffer to flush
    while exit_time.elapsed() < FLUSH_TIMEOUT {
        vex_sdk::vexTasksRun();

        // If the buffer has been fully flushed, exit the loop
        if vex_sdk::vexSerialWriteFree(stdio::STDIO_CHANNEL) == (stdio::STDOUT_BUF_SIZE as i32) {
            break;
        }
    }
}

pub fn abort_internal() -> ! {
    unsafe {
        vex_sdk::vexSystemExitRequest();

        loop {
            vex_sdk::vexTasksRun();
        }
    }
}
