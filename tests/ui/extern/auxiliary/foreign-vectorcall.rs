// Make the aux build not use `dylib` to avoid https://github.com/rust-lang/rust/issues/130196
//@ no-prefer-dynamic
#![feature(abi_vectorcall)]
#![crate_type = "lib"]

#[no_mangle]
#[inline(never)]
pub extern "vectorcall" fn call_with_42(i: i32) {
    assert_eq!(i, 42);
}
