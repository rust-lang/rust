#![feature(lang_items)]
#![no_std]

#[panic_handler]
pub fn begin_panic_handler(_info: &core::panic::PanicInfo<'_>) -> ! {
    loop {}
}

#[lang = "eh_personality"]
extern "C" fn eh_personality() {}
