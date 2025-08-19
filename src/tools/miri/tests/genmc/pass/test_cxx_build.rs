//@compile-flags: -Zmiri-genmc

#![no_main]

#[unsafe(no_mangle)]
fn miri_start(_argc: isize, _argv: *const *const u8) -> isize {
    0
}
