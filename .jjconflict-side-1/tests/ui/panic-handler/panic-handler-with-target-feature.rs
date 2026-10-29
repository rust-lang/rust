//@ compile-flags:-C panic=abort
//@ only-x86_64

#![no_std]
#![no_main]

use core::panic::PanicInfo;

#[panic_handler]
#[target_feature(enable = "avx2")]
//~^ ERROR `#[target_feature]` cannot be applied to a `#[panic_handler]` function
fn panic(info: &PanicInfo) -> ! {
    unimplemented!();
}
