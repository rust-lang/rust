#![feature(lang_items)]
#![no_std]

// Since `rustc` generally passes `-nodefaultlibs` to the linker,
// Rust programs link necessary system libraries via `#[link()]`
// attributes in the `libc` crate. `libc` is a dependency of `std`,
// but as we are `#![no_std]`, we need to include it manually.
#![feature(rustc_private)]
extern crate libc;

#[panic_handler]
pub fn begin_panic_handler(_info: &core::panic::PanicInfo<'_>) -> ! {
    loop {}
}

#[lang = "eh_personality"]
extern "C" fn eh_personality() {}
