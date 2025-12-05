#![crate_type = "bin"]
#![feature(lang_items)]
#![no_main]
#![no_std]

use core::panic::PanicInfo;

const Z: () = panic!("cheese");
//~^ ERROR evaluation panicked

const Y: () = unreachable!();
//~^ ERROR evaluation panicked

const X: () = unimplemented!();
//~^ ERROR evaluation panicked

#[lang = "eh_personality"]
fn eh() {}
#[lang = "eh_catch_typeinfo"]
static EH_CATCH_TYPEINFO: u8 = 0;

#[panic_handler]
fn panic(_info: &PanicInfo) -> ! {
    loop {}
}
