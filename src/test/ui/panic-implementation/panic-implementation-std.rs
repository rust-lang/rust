// error-pattern: duplicate lang item found: `panic_impl`.

#![feature(panic_implementation)]

use std::panic::PanicInfo;

#[panic_implementation]
fn panic(info: PanicInfo) -> ! {
    loop {}
}

fn main() {}
