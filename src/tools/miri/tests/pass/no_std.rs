#![feature(lang_items, start)]
#![no_std]
// windows tls dtors go through libstd right now, thus this test
// cannot pass. When windows tls dtors go through the special magic
// windows linker section, we can run this test on windows again.
//@ignore-target-windows

#[start]
fn start(_: isize, _: *const *const u8) -> isize {
    for _ in 0..10 {}

    0
}

#[panic_handler]
fn panic_handler(_: &core::panic::PanicInfo) -> ! {
    loop {}
}

#[lang = "eh_personality"]
fn eh_personality() {}
