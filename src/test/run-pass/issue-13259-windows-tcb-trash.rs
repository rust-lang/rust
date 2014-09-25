extern crate libc;
use libc::{c_void, LPVOID, DWORD};
use libc::types::os::arch::extra::LPWSTR;

extern "system" {
    fn FormatMessageW(flags: DWORD,
                      lpSrc: LPVOID,
                      msgId: DWORD,
                      langId: DWORD,
                      buf: LPWSTR,
                      nsize: DWORD,
                      args: *const c_void)
                      -> DWORD;
}

fn test() {
    let mut buf: [u16, ..50] = [0, ..50];
    let ret = unsafe {
        FormatMessageW(0x1000, 0 as *mut c_void, 1, 0x400,
                       buf.as_mut_ptr(), buf.len() as u32, 0 as *const c_void)
    };
    // On some 32-bit Windowses (Win7-8 at least) this will fail with segmented
    // stacks taking control of pvArbitrary
    assert!(ret != 0);
}
fn main() {
    test()
}