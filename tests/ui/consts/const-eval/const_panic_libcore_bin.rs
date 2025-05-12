#![crate_type = "bin"]
#![feature(lang_items)]
#![no_main]
#![no_std]

use core::panic::PanicInfo;

const Z: () = panic!("cheese");
//~^ ERROR evaluation of constant value failed

const Y: () = unreachable!();
//~^ ERROR evaluation of constant value failed

const X: () = unimplemented!();
//~^ ERROR evaluation of constant value failed

#[lang = "eh_personality"]
fn eh() {}
#[lang = "eh_catch_typeinfo"]
static EH_CATCH_TYPEINFO: u8 = 0;

#[panic_handler]
fn panic(_info: &PanicInfo) -> ! {
    loop {}
}
