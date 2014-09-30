// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

extern crate libc;

#[cfg(windows)]
mod imp {
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

    pub fn test() {
        let mut buf: [u16, ..50] = [0, ..50];
        let ret = unsafe {
            FormatMessageW(0x1000, 0 as *mut c_void, 1, 0x400,
                           buf.as_mut_ptr(), buf.len() as u32, 0 as *const c_void)
        };
        // On some 32-bit Windowses (Win7-8 at least) this will fail with segmented
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
