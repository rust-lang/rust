//! Regression test for #98254, missing `__chkstk` symbol on `aarch64-unknown-uefi`.
//@ build-pass
//@ only-aarch64-unknown-uefi
//@ compile-flags: -Cpanic=abort
//@ compile-flags: -Clinker=rust-lld
#![no_std]
#![no_main]
#[panic_handler]
fn panic_handler(_info: &core::panic::PanicInfo) -> ! {
    loop {}
}

#[export_name = "efi_main"]
fn main() {
    let b = [0; 1024];
}
