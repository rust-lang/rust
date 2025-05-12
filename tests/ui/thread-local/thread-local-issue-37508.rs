//@ only-x86_64
//@ compile-flags: -Ccode-model=large --crate-type lib
//@ build-pass
//
// Regression test for issue #37508

// FIXME(static_mut_refs): Do not allow `static_mut_refs` lint
#![allow(static_mut_refs)]

#![no_main]
#![no_std]
#![feature(thread_local, lang_items)]

#[lang = "eh_personality"]
extern "C" fn eh_personality() {}

use core::panic::PanicInfo;

#[panic_handler]
fn panic(_panic: &PanicInfo<'_>) -> ! {
    loop {}
}

pub struct BB;

#[thread_local]
static mut KEY: Key = Key { inner: BB, dtor_running: false };

pub unsafe fn set() -> Option<&'static BB> {
    if KEY.dtor_running {
        return None;
    }
    Some(&KEY.inner)
}

pub struct Key {
    inner: BB,
    dtor_running: bool,
}
