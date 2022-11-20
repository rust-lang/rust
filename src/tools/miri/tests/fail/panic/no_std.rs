#![feature(lang_items, start, core_intrinsics)]
#![no_std]
// windows tls dtors go through libstd right now, thus this test
// cannot pass. When windows tls dtors go through the special magic
// windows linker section, we can run this test on windows again.
//@ignore-target-windows: no-std not supported on Windows

// Plumbing to let us use `writeln!` to host stderr:

extern "Rust" {
    fn miri_write_to_stderr(bytes: &[u8]);
}

struct HostErr;

use core::fmt::Write;

impl Write for HostErr {
    fn write_str(&mut self, s: &str) -> core::fmt::Result {
        unsafe {
            miri_write_to_stderr(s.as_bytes());
        }
        Ok(())
    }
}

// Aaaand the test:

#[start]
fn start(_: isize, _: *const *const u8) -> isize {
    panic!("blarg I am dead")
}

#[panic_handler]
fn panic_handler(panic_info: &core::panic::PanicInfo) -> ! {
    writeln!(HostErr, "{panic_info}").ok();
    core::intrinsics::abort(); //~ ERROR: the program aborted execution
}

#[lang = "eh_personality"]
fn eh_personality() {}
