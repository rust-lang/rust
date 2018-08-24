// compile-flags:-C panic=abort

#![no_std]
#![no_main]

use core::panic::PanicInfo;

#[panic_implementation] //~ ERROR #[panic_implementation] is an unstable feature (see issue #44489)
fn panic(info: &PanicInfo) -> ! {
    loop {}
}
