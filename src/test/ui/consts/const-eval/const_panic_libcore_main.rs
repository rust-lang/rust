#![crate_type = "bin"]
#![feature(lang_items)]
#![feature(panic_implementation)]
#![feature(const_panic)]
#![no_main]
#![no_std]

use core::panic::PanicInfo;

const Z: () = panic!("cheese");
//~^ ERROR this constant cannot be used

const Y: () = unreachable!();
//~^ ERROR this constant cannot be used

const X: () = unimplemented!();
//~^ ERROR this constant cannot be used

#[lang = "eh_personality"]
fn eh() {}
#[lang = "eh_unwind_resume"]
fn eh_unwind_resume() {}

#[panic_implementation]
fn panic(_info: &PanicInfo) -> ! {
    loop {}
}
