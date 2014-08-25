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
    use rt::stack;
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
    static NTE_BAD_SIGNATURE: DWORD = 0x80090006;

    #[allow(non_snake_case_functions)]
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
            let mut ret = unsafe {
                CryptAcquireContextA(&mut hcp, 0 as LPCSTR, 0 as LPCSTR,
                                     PROV_RSA_FULL,
                                     CRYPT_VERIFYCONTEXT | CRYPT_SILENT)
            };

            // FIXME #13259:
            // It turns out that if we can't acquire a context with the
            // NTE_BAD_SIGNATURE error code, the documentation states:
            //
            //     The provider DLL signature could not be verified. Either the
            //     DLL or the digital signature has been tampered with.
            //
            // Sounds fishy, no? As it turns out, our signature can be bad
            // because our Thread Information Block (TIB) isn't exactly what it
            // expects. As to why, I have no idea. The only data we store in the
            // TIB is the stack limit for each thread, but apparently that's
            // enough to make the signature valid.
            //
            // Furthermore, this error only happens the *first* time we call
            // CryptAcquireContext, so we don't have to worry about future
            // calls.
            //
            // Anyway, the fix employed here is that if we see this error, we
            // pray that we're not close to the end of the stack, temporarily
            // set the stack limit to 0 (what the TIB originally was), acquire a
            // context, and then reset the stack limit.
            //
            // Again, I'm not sure why this is the fix, nor why we're getting
            // this error. All I can say is that this seems to allow libnative
            // to progress where it otherwise would be hindered. Who knew?
            if ret == 0 && os::errno() as DWORD == NTE_BAD_SIGNATURE {
                unsafe {
                    let limit = stack::get_sp_limit();
                    stack::record_sp_limit(0);
                    ret = CryptAcquireContextA(&mut hcp, 0 as LPCSTR, 0 as LPCSTR,
                                               PROV_RSA_FULL,
                                               CRYPT_VERIFYCONTEXT | CRYPT_SILENT);
                    stack::record_sp_limit(limit);
                }
            }

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
