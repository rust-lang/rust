#![feature(lang_items, panic_unwind)]
#![no_std]

// Since the `unwind` crate is a dependency of the `std` crate, and we have
// `#![no_std]`, the unwinder is not included in the link command by default.
// We need to include crate `unwind` manually.
extern crate unwind;

#[panic_handler]
pub fn begin_panic_handler(_info: &core::panic::PanicInfo<'_>) -> ! {
    loop {}
}

#[lang = "eh_personality"]
extern "C" fn eh_personality(
    _version: i32,
    _actions: i32,
    _exception_class: u64,
    _exception_object: *mut (),
    _context: *mut (),
) -> i32 {
    loop {}
}
