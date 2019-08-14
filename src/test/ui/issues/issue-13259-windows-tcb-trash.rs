// run-pass
#![feature(rustc_private)]

extern crate libc;

#[cfg(windows)]
mod imp {
    type LPVOID = *mut u8;
    type DWORD = u32;
    type LPWSTR = *mut u16;

    extern "system" {
        fn FormatMessageW(flags: DWORD,
                          lpSrc: LPVOID,
                          msgId: DWORD,
                          langId: DWORD,
                          buf: LPWSTR,
                          nsize: DWORD,
                          args: *const u8)
                          -> DWORD;
    }

    pub fn test() {
        let mut buf: [u16; 50] = [0; 50];
        let ret = unsafe {
            FormatMessageW(0x1000, core::ptr::null_mut(), 1, 0x400,
                           buf.as_mut_ptr(), buf.len() as u32, core::ptr::null())
        };
        // On some 32-bit Windowses (Win7-8 at least) this will panic with segmented
        // stacks taking control of pvArbitrary
        assert!(ret != 0);
    }
}

#[cfg(not(windows))]
mod imp {
    pub fn test() { }
}

fn main() {
    imp::test()
}
