#![crate_type="staticlib"]

#[no_mangle]
pub extern "C" fn rust_always_inlined() -> u32 {
    42
}

#[no_mangle]
#[inline(never)]
pub extern "C" fn rust_never_inlined() -> u32 {
    421
}
