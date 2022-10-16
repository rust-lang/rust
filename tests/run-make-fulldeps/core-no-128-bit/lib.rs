#![feature(no_core)]

#![no_std]
#![no_core] // supress compiler-builtins
extern crate core;
use core::prelude::rust_2021::*;
use core::fmt::Write;

// An empty file might be sufficient here, but since formatting is one of the
// features affected by no_128_bit it seems worth including some.

struct X(pub usize);
impl core::fmt::Write for X {
    fn write_str(&mut self, s: &str) -> core::fmt::Result {
        self.0 += s.len();
        Ok(())
    }
}

#[no_mangle]
extern "C" fn demo() -> usize {
    let mut x = X(0);
    // Writes "i128 u128 foo" due to the removal of u/i128 formatting.
    core::write!(x, "{:?} {:?} {}", i128::MAX, u128::MIN, "foo").unwrap();
    x.0
}

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    loop {}
}
