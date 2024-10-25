#[path = "../unsupported/args.rs"]
pub mod args;
pub mod env;
pub mod fs;
#[path = "../unsupported/io.rs"]
pub mod io;
#[path = "../unsupported/net.rs"]
pub mod net;
#[path = "../unsupported/os.rs"]
pub mod os;
#[path = "../unsupported/pipe.rs"]
pub mod pipe;
#[path = "../unsupported/process.rs"]
pub mod process;
pub mod stdio;
#[path = "../unsupported/thread.rs"]
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

    // VEXos doesn't explicitly clean out .bss.
    ptr::slice_from_raw_parts_mut(
        addr_of_mut!(__bss_start),
        addr_of_mut!(__bss_end).offset_from(addr_of_mut!(__bss_start)) as usize,
    )
    .as_mut()
    .unwrap_unchecked()
    .fill(0);

    main();

    abort_internal()
}

// The code signature is a 32 byte header at the start of user programs that
// identifies the owner and type of the program, as well as certain flags for
// program behavior dictated by the OS. In the event that the user wants to
// change this header, we use weak linkage so it can be overwritten.
#[link_section = ".code_signature"]
#[linkage = "weak"]
#[used]
static CODE_SIGNATURE: vex_sdk::vcodesig =
    vex_sdk::vcodesig { magic: u32::from_le_bytes(*b"XVX5"), r#type: 0, owner: 2, options: 0 };

// This function is needed by the panic runtime. The symbol is named in
// pre-link args for the target specification, so keep that in sync.
#[cfg(not(test))]
#[no_mangle]
// NB. used by both libunwind and libpanic_abort
pub extern "C" fn __rust_abort() -> ! {
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
