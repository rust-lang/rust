// ignore-tidy-linelength
//@ build-pass
//@ only-x86
//@ only-windows
//@ ignore-gnu - vectorcall is not supported by GCC: https://gcc.gnu.org/bugzilla/show_bug.cgi?id=89485
#![crate_type = "cdylib"]
#![feature(abi_vectorcall)]

#[no_mangle]
extern "stdcall" fn foo(_: bool) {}

#[no_mangle]
extern "fastcall" fn bar(_: u8) {}

#[no_mangle]
extern "vectorcall" fn baz(_: u16) {}
