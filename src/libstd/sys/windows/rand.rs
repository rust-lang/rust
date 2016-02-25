// Copyright 2013-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


use io;
use mem;
use rand::Rng;
use sys::c;

pub struct OsRng {
    hcryptprov: c::HCRYPTPROV
}

impl OsRng {
    /// Create a new `OsRng`.
    pub fn new() -> io::Result<OsRng> {
        let mut hcp = 0;
        let ret = unsafe {
            c::CryptAcquireContextA(&mut hcp, 0 as c::LPCSTR, 0 as c::LPCSTR,
                                    c::PROV_RSA_FULL,
                                    c::CRYPT_VERIFYCONTEXT | c::CRYPT_SILENT)
        };

        if ret == 0 {
            Err(io::Error::last_os_error())
        } else {
            Ok(OsRng { hcryptprov: hcp })
        }
    }
}

impl Rng for OsRng {
    fn next_u32(&mut self) -> u32 {
        let mut v = [0; 4];
        self.fill_bytes(&mut v);
        unsafe { mem::transmute(v) }
    }
    fn next_u64(&mut self) -> u64 {
        let mut v = [0; 8];
        self.fill_bytes(&mut v);
        unsafe { mem::transmute(v) }
    }
    fn fill_bytes(&mut self, v: &mut [u8]) {
        // CryptGenRandom takes a DWORD (u32) for the length so we need to
        // split up the buffer.
        for slice in v.chunks_mut(<c::DWORD>::max_value() as usize) {
            let ret = unsafe {
                c::CryptGenRandom(self.hcryptprov, slice.len() as c::DWORD,
                                  slice.as_mut_ptr())
            };
            if ret == 0 {
                panic!("couldn't generate random bytes: {}",
                       io::Error::last_os_error());
            }
        }
    }
}

impl Drop for OsRng {
    fn drop(&mut self) {
        let ret = unsafe {
            c::CryptReleaseContext(self.hcryptprov, 0)
        };
        if ret == 0 {
            panic!("couldn't release context: {}",
                   io::Error::last_os_error());
        }
    }
}
