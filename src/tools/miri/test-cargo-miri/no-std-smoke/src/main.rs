// Copied from tests/pass/no-std.rs

#![no_std]
#![no_main]

// Plumbing to let us use `writeln!` to host stdout:

extern "Rust" {
    fn miri_write_to_stdout(bytes: &[u8]);
}

struct Host;

use core::fmt::Write;

impl Write for Host {
    fn write_str(&mut self, s: &str) -> core::fmt::Result {
        unsafe {
            miri_write_to_stdout(s.as_bytes());
        }
        Ok(())
    }
}

#[no_mangle]
fn miri_start(_: isize, _: *const *const u8) -> isize {
    writeln!(Host, "hello, world!").unwrap();
    0
}

#[panic_handler]
fn panic_handler(_: &core::panic::PanicInfo) -> ! {
    loop {}
}
