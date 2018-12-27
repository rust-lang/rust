#![crate_type = "bin"]
#![feature(lang_items)]
#![feature(const_panic)]
#![no_main]
#![no_std]

use core::panic::PanicInfo;

const Z: () = panic!("cheese");
//~^ ERROR any use of this value will cause an error

const Y: () = unreachable!();
//~^ ERROR any use of this value will cause an error

const X: () = unimplemented!();
//~^ ERROR any use of this value will cause an error

#[lang = "eh_personality"]
fn eh() {}
#[lang = "eh_unwind_resume"]
fn eh_unwind_resume() {}

#[panic_handler]
fn panic(_info: &PanicInfo) -> ! {
    loop {}
}
