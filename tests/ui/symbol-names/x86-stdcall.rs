// build-pass
// only-x86-windows
#![crate_type = "cdylib"]
#![feature(abi_vectorcall)]

#[no_mangle]
extern "stdcall" fn foo(_: bool) {}

#[no_mangle]
extern "fastcall" fn bar(_: u8) {}

#[no_mangle]
extern "vectorcall" fn baz(_: u16) {}
