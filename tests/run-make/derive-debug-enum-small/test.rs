// regression test for https://github.com/rust-lang/rust/issues/165189
// Based on the snippet shared by @folkertdev

#![no_std]
#![no_main]

use core::fmt::Write;

#[derive(Debug)]
pub enum Error {
    Overrun,
    Framing,
    Parity,
    Noise,
    Timeout,
}

struct Sink;

impl Write for Sink {
    fn write_str(&mut self, s: &str) -> core::fmt::Result {
        for b in s.bytes() {
            unsafe { core::ptr::write_volatile(0x4000_0000 as *mut u8, b) };
        }
        Ok(())
    }
}

#[panic_handler]
fn panic(info: &core::panic::PanicInfo) -> ! {
    let _ = write!(Sink, "{}", info.message());
    loop {}
}

#[unsafe(no_mangle)]
extern "C" fn rust_eh_personality() {}

fn read() -> Result<u8, Error> {
    match unsafe { core::ptr::read_volatile(0x4000_0004 as *const u8) } {
        0 => Err(Error::Overrun),
        1 => Err(Error::Framing),
        2 => Err(Error::Parity),
        3 => Err(Error::Noise),
        4 => Err(Error::Timeout),
        b => Ok(b),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn _start() -> ! {
    loop {
        let b = read().unwrap();
        unsafe { core::ptr::write_volatile(0x4000_0000 as *mut u8, b) };
    }
}
