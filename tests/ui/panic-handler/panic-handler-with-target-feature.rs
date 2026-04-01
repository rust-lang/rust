//@ compile-flags:-C panic=abort
//@ only-x86_64

#![no_std]
#![no_main]

use core::panic::PanicInfo;

#[panic_handler]
#[target_feature(enable = "avx2")]
//~^ ERROR `#[panic_handler]` function is not allowed to have `#[target_feature]`
fn panic(info: &PanicInfo) -> ! {
    unimplemented!();
}
