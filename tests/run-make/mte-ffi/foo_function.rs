#![crate_type = "cdylib"]
#![crate_name = "foo"]

extern "C" fn ret32() -> i32 {
    32
}

#[no_mangle]
pub extern "C" fn foo(ptr: extern "C" fn(extern "C" fn() -> i32)) {
    assert_eq!((ptr as usize) >> 56, 0x1f);

    // Store an arbitrary tag in the tag bits, and convert back to the correct pointer type.
    let p = ((ret32 as usize) | (0x2f << 56)) as *const ();
    let p: extern "C" fn() -> i32 = unsafe { std::mem::transmute(p) };

    unsafe { ptr(p) }
}
