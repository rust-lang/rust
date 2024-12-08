#![crate_type = "cdylib"]
#![crate_name = "foo"]

use std::arch::asm;
use std::ffi::{CStr, CString};
use std::os::raw::c_char;

extern "C" {
    fn bar(ptr: *const c_char);
}

#[no_mangle]
pub extern "C" fn foo(ptr: *const c_char) {
    assert_eq!((ptr as usize) >> 56, 0x1f);

    let s = unsafe { CStr::from_ptr(ptr) };
    assert_eq!(s.to_str().unwrap(), "ab");

    let s = CString::from_vec_with_nul("cd\0".into()).unwrap();
    let mut p = ((s.as_ptr() as usize) | (0x2f << 56)) as *const c_char;
    unsafe {
        #[cfg(target_feature = "mte")]
        asm!("stg {p}, [{p}]", p = inout(reg) p);

        bar(p);
    }
}
