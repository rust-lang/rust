//@ only-64bit
//@ edition: 2024
//@ build-fail

#![crate_type = "lib"]

#[repr(C)]
pub struct Big([u8; 1 << 63]);

#[unsafe(no_mangle)]
pub extern "C" fn foo(_x: Big) {}
//~^ ERROR values of the type `[u8; 9223372036854775808]` are too big for the target architecture
