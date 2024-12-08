//@ edition:2018
//
// This is a regression test for #83564.
// For some reason, Rust 2018 or higher is required to reproduce the bug.
//@ run-rustfix
//@ revisions: no_std std
//@ [no_std]compile-flags: -C panic=abort
#![cfg_attr(no_std, no_std)]

fn main() {
    //~^ HELP consider importing this struct
    let _x = NonZero::new(5u32).unwrap();
    //~^ ERROR failed to resolve: use of undeclared type `NonZero`
}

#[allow(dead_code)]
#[cfg_attr(no_std, panic_handler)]
fn panic(_info: &core::panic::PanicInfo) -> ! {
    loop {}
}
