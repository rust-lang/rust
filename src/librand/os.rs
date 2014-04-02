// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
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

pub use self::imp::OSRng;

#[cfg(unix)]
mod imp {
    use Rng;
    use reader::ReaderRng;
    use std::io::{IoResult, File};

    /// A random number generator that retrieves randomness straight from
    /// the operating system. Platform sources:
    ///
    /// - Unix-like systems (Linux, Android, Mac OSX): read directly from
    ///   `/dev/urandom`.
    /// - Windows: calls `CryptGenRandom`, using the default cryptographic
    ///   service provider with the `PROV_RSA_FULL` type.
    ///
    /// This does not block.
    #[cfg(unix)]
    pub struct OSRng {
        inner: ReaderRng<File>
    }

    impl OSRng {
        /// Create a new `OSRng`.
        pub fn new() -> IoResult<OSRng> {
            let reader = try!(File::open(&Path::new("/dev/urandom")));
            let reader_rng = ReaderRng::new(reader);

            Ok(OSRng { inner: reader_rng })
        }
    }

    impl Rng for OSRng {
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

#[cfg(windows)]
mod imp {
    use Rng;
    use std::cast;
    use std::io::{IoResult, IoError};
    use std::libc::{c_ulong, DWORD, BYTE, LPCSTR, BOOL};
    use std::os;
    use std::rt::stack;

    type HCRYPTPROV = c_ulong;

    /// A random number generator that retrieves randomness straight from
    /// the operating system. Platform sources:
    ///
    /// - Unix-like systems (Linux, Android, Mac OSX): read directly from
    ///   `/dev/urandom`.
    /// - Windows: calls `CryptGenRandom`, using the default cryptographic
    ///   service provider with the `PROV_RSA_FULL` type.
    ///
    /// This does not block.
    pub struct OSRng {
        hcryptprov: HCRYPTPROV
    }

    static PROV_RSA_FULL: DWORD = 1;
    static CRYPT_SILENT: DWORD = 64;
    static CRYPT_VERIFYCONTEXT: DWORD = 0xF0000000;
    static NTE_BAD_SIGNATURE: DWORD = 0x80090006;

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

    impl OSRng {
        /// Create a new `OSRng`.
        pub fn new() -> IoResult<OSRng> {
            let mut hcp = 0;
            let mut ret = unsafe {
                CryptAcquireContextA(&mut hcp, 0 as LPCSTR, 0 as LPCSTR,
                                     PROV_RSA_FULL,
                                     CRYPT_VERIFYCONTEXT | CRYPT_SILENT)
            };

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
                Ok(OSRng { hcryptprov: hcp })
            }
        }
    }

    impl Rng for OSRng {
        fn next_u32(&mut self) -> u32 {
            let mut v = [0u8, .. 4];
            self.fill_bytes(v);
            unsafe { cast::transmute(v) }
        }
        fn next_u64(&mut self) -> u64 {
            let mut v = [0u8, .. 8];
            self.fill_bytes(v);
            unsafe { cast::transmute(v) }
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

    impl Drop for OSRng {
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
    use super::OSRng;
    use Rng;
    use std::task;

    #[test]
    fn test_os_rng() {
        let mut r = OSRng::new().unwrap();

        r.next_u32();
        r.next_u64();

        let mut v = [0u8, .. 1000];
        r.fill_bytes(v);
    }

    #[test]
    fn test_os_rng_tasks() {

        let mut txs = vec!();
        for _ in range(0, 20) {
            let (tx, rx) = channel();
            txs.push(tx);
            task::spawn(proc() {
                // wait until all the tasks are ready to go.
                rx.recv();

                // deschedule to attempt to interleave things as much
                // as possible (XXX: is this a good test?)
                let mut r = OSRng::new().unwrap();
                task::deschedule();
                let mut v = [0u8, .. 1000];

                for _ in range(0, 100) {
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
