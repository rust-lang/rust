#[expect(dead_code)]
#[path = "../unsupported/common.rs"]
mod unsupported_common;

pub use unsupported_common::{init, unsupported, unsupported_err};

use crate::arch::global_asm;
use crate::sys::stdio;
use crate::time::{Duration, Instant};

global_asm!(
    r#"
    .section .boot, "ax"
    .global _boot

    .arm
    _boot:
        @ Set up the user stack.
        ldr sp, =__stack_top

        @ Clear the .bss (uninitialized statics) section by filling it with zeroes.
        @ This is required, since the compiler assumes it will be zeroed on first access.
        mov r0, #0
        ldr r1, =__bss_start
        ldr r2, =__bss_end
    .Lclear_bss:
        cmp r1, r2
        beq .Lbss_done
        str r0, [r1], #4
        b .Lclear_bss
    .Lbss_done:
        blx _start @ Jump to the Rust entrypoint.
    "#
);

#[cfg(not(test))]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn _start() -> ! {
    unsafe extern "C" {
        fn main() -> i32;
    }

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
