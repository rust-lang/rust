//@only-target: windows # this directly tests windows-only functions

use std::ffi::c_void;
use std::ptr;

extern "system" {
    fn TlsAlloc() -> u32;
    fn TlsSetValue(key: u32, val: *mut c_void) -> bool;
    fn TlsGetValue(key: u32) -> *mut c_void;
    fn TlsFree(key: u32) -> bool;
}

fn main() {
    let key = unsafe { TlsAlloc() };
    assert!(unsafe { TlsSetValue(key, ptr::without_provenance_mut(1)) });
    assert_eq!(unsafe { TlsGetValue(key).addr() }, 1);
    assert!(unsafe { TlsFree(key) });
}
