#![crate_type = "cdylib"]
#![crate_name = "foo"]

use std::os::raw::c_float;

extern "C" {
    fn bar(ptr: *const c_float);
}

#[no_mangle]
pub extern "C" fn foo(ptr: *mut c_float) {
    assert_eq!((ptr as usize) >> 56, 0x1f);

    unsafe {
        *ptr = 0.5;
        *ptr.wrapping_add(1) = 0.2;
        bar(ptr);
    }
}
