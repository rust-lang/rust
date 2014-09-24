// Copyright 2013-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Interfaces to the operating system provided random number
//! generators.

pub use self::imp::OsRng;

#[cfg(unix, not(target_os = "ios"))]
mod imp {
    use io::{IoResult, File};
    use path::Path;
    use rand::Rng;
    use rand::reader::ReaderRng;
    use result::{Ok, Err};

    /// A random number generator that retrieves randomness straight from
    /// the operating system. Platform sources:
    ///
    /// - Unix-like systems (Linux, Android, Mac OSX): read directly from
    ///   `/dev/urandom`.
    /// - Windows: calls `CryptGenRandom`, using the default cryptographic
    ///   service provider with the `PROV_RSA_FULL` type.
    /// - iOS: calls SecRandomCopyBytes as /dev/(u)random is sandboxed
    /// This does not block.
    #[cfg(unix)]
    pub struct OsRng {
        inner: ReaderRng<File>
    }

    impl OsRng {
        /// Create a new `OsRng`.
        pub fn new() -> IoResult<OsRng> {
            let reader = try!(File::open(&Path::new("/dev/urandom")));
            let reader_rng = ReaderRng::new(reader);

            Ok(OsRng { inner: reader_rng })
        }
    }

    impl Rng for OsRng {
        fn next_u32(&mut self) -> u32 {
            self.inner.next_u32()
        }
        fn next_u64(&mut self) -> u64 {
            self.inner.next_u64()
        }
        fn fill_bytes(&mut self, v: &mut [u8]) {
            self.inner.fill_bytes(v)
        }
    }
}

#[cfg(target_os = "ios")]
mod imp {
    extern crate libc;

    use collections::Collection;
    use io::{IoResult};
    use kinds::marker;
    use mem;
    use os;
    use rand::Rng;
    use result::{Ok};
    use self::libc::{c_int, size_t};
    use slice::MutableSlice;

    /// A random number generator that retrieves randomness straight from
    /// the operating system. Platform sources:
    ///
    /// - Unix-like systems (Linux, Android, Mac OSX): read directly from
    ///   `/dev/urandom`.
    /// - Windows: calls `CryptGenRandom`, using the default cryptographic
    ///   service provider with the `PROV_RSA_FULL` type.
    /// - iOS: calls SecRandomCopyBytes as /dev/(u)random is sandboxed
    /// This does not block.
    pub struct OsRng {
        marker: marker::NoCopy
    }

    #[repr(C)]
    struct SecRandom;

    static kSecRandomDefault: *const SecRandom = 0 as *const SecRandom;

    #[link(name = "Security", kind = "framework")]
    extern "C" {
        fn SecRandomCopyBytes(rnd: *const SecRandom,
                              count: size_t, bytes: *mut u8) -> c_int;
    }

    impl OsRng {
        /// Create a new `OsRng`.
        pub fn new() -> IoResult<OsRng> {
            Ok(OsRng {marker: marker::NoCopy} )
        }
    }

    impl Rng for OsRng {
        fn next_u32(&mut self) -> u32 {
            let mut v = [0u8, .. 4];
            self.fill_bytes(v);
            unsafe { mem::transmute(v) }
        }
        fn next_u64(&mut self) -> u64 {
            let mut v = [0u8, .. 8];
            self.fill_bytes(v);
            unsafe { mem::transmute(v) }
        }
        fn fill_bytes(&mut self, v: &mut [u8]) {
            let ret = unsafe {
                SecRandomCopyBytes(kSecRandomDefault, v.len() as size_t, v.as_mut_ptr())
            };
            if ret == -1 {
                fail!("couldn't generate random bytes: {}", os::last_os_error());
            }
        }
    }
}

#[cfg(windows)]
mod imp {
    extern crate libc;

    use core_collections::Collection;
    use io::{IoResult, IoError};
    use mem;
    use ops::Drop;
    use os;
    use rand::Rng;
    use result::{Ok, Err};
    use self::libc::{DWORD, BYTE, LPCSTR, BOOL};
    use self::libc::types::os::arch::extra::{LONG_PTR};
    use slice::MutableSlice;

    type HCRYPTPROV = LONG_PTR;

    /// A random number generator that retrieves randomness straight from
    /// the operating system. Platform sources:
    ///
    /// - Unix-like systems (Linux, Android, Mac OSX): read directly from
    ///   `/dev/urandom`.
    /// - Windows: calls `CryptGenRandom`, using the default cryptographic
    ///   service provider with the `PROV_RSA_FULL` type.
    ///
    /// This does not block.
    pub struct OsRng {
        hcryptprov: HCRYPTPROV
    }

    static PROV_RSA_FULL: DWORD = 1;
    static CRYPT_SILENT: DWORD = 64;
    static CRYPT_VERIFYCONTEXT: DWORD = 0xF0000000;

    #[allow(non_snake_case)]
    extern "system" {
        fn CryptAcquireContextA(phProv: *mut HCRYPTPROV,
                                pszContainer: LPCSTR,
                                pszProvider: LPCSTR,
                                dwProvType: DWORD,
                                dwFlags: DWORD) -> BOOL;
        fn CryptGenRandom(hProv: HCRYPTPROV,
                          dwLen: DWORD,
                          pbBuffer: *mut BYTE) -> BOOL;
        fn CryptReleaseContext(hProv: HCRYPTPROV, dwFlags: DWORD) -> BOOL;
    }

    impl OsRng {
        /// Create a new `OsRng`.
        pub fn new() -> IoResult<OsRng> {
            let mut hcp = 0;
            let ret = unsafe {
                CryptAcquireContextA(&mut hcp, 0 as LPCSTR, 0 as LPCSTR,
                                     PROV_RSA_FULL,
                                     CRYPT_VERIFYCONTEXT | CRYPT_SILENT)
            };

            if ret == 0 {
                Err(IoError::last_error())
            } else {
                Ok(OsRng { hcryptprov: hcp })
            }
        }
    }

    impl Rng for OsRng {
        fn next_u32(&mut self) -> u32 {
            let mut v = [0u8, .. 4];
            self.fill_bytes(v);
            unsafe { mem::transmute(v) }
        }
        fn next_u64(&mut self) -> u64 {
            let mut v = [0u8, .. 8];
            self.fill_bytes(v);
            unsafe { mem::transmute(v) }
        }
        fn fill_bytes(&mut self, v: &mut [u8]) {
            let ret = unsafe {
                CryptGenRandom(self.hcryptprov, v.len() as DWORD,
                               v.as_mut_ptr())
            };
            if ret == 0 {
                fail!("couldn't generate random bytes: {}", os::last_os_error());
            }
        }
    }

    impl Drop for OsRng {
        fn drop(&mut self) {
            let ret = unsafe {
                CryptReleaseContext(self.hcryptprov, 0)
            };
            if ret == 0 {
                fail!("couldn't release context: {}", os::last_os_error());
            }
        }
    }
}

#[cfg(test)]
mod test {
    use prelude::*;

    use super::OsRng;
    use rand::Rng;
    use task;

    #[test]
    fn test_os_rng() {
        let mut r = OsRng::new().unwrap();

        r.next_u32();
        r.next_u64();

        let mut v = [0u8, .. 1000];
        r.fill_bytes(v);
    }

    #[test]
    fn test_os_rng_tasks() {

        let mut txs = vec!();
        for _ in range(0u, 20) {
            let (tx, rx) = channel();
            txs.push(tx);
            task::spawn(proc() {
                // wait until all the tasks are ready to go.
                rx.recv();

                // deschedule to attempt to interleave things as much
                // as possible (XXX: is this a good test?)
                let mut r = OsRng::new().unwrap();
                task::deschedule();
                let mut v = [0u8, .. 1000];

                for _ in range(0u, 100) {
                    r.next_u32();
                    task::deschedule();
                    r.next_u64();
                    task::deschedule();
                    r.fill_bytes(v);
                    task::deschedule();
                }
            })
        }

        // start all the tasks
        for tx in txs.iter() {
            tx.send(())
        }
    }
}
