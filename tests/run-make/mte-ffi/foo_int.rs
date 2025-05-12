#![crate_type = "cdylib"]
#![crate_name = "foo"]

use std::os::raw::c_uint;

extern "C" {
    fn bar(ptr: *const c_uint);
}

#[no_mangle]
pub extern "C" fn foo(ptr: *mut c_uint) {
    assert_eq!((ptr as usize) >> 56, 0x1f);

    unsafe {
        *ptr = 0x63;
        *ptr.wrapping_add(1) = 0x64;
        bar(ptr);
    }
}
