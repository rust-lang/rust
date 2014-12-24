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

#[cfg(all(unix, not(target_os = "ios")))]
mod imp {
    extern crate libc;

    use self::OsRngInner::*;

    use io::{IoResult, File};
    use path::Path;
    use rand::Rng;
    use rand::reader::ReaderRng;
    use result::Result::{Ok, Err};
    use slice::SliceExt;
    use mem;
    use os::errno;

    #[cfg(all(target_os = "linux",
              any(target_arch = "x86_64", target_arch = "x86", target_arch = "arm")))]
    fn getrandom(buf: &mut [u8]) -> libc::c_long {
        extern "C" {
            fn syscall(number: libc::c_long, ...) -> libc::c_long;
        }

        #[cfg(target_arch = "x86_64")]
        const NR_GETRANDOM: libc::c_long = 318;
        #[cfg(target_arch = "x86")]
        const NR_GETRANDOM: libc::c_long = 355;
        #[cfg(target_arch = "arm")]
        const NR_GETRANDOM: libc::c_long = 384;

        unsafe {
            syscall(NR_GETRANDOM, buf.as_mut_ptr(), buf.len(), 0u)
        }
    }

    #[cfg(not(all(target_os = "linux",
                  any(target_arch = "x86_64", target_arch = "x86", target_arch = "arm"))))]
    fn getrandom(_buf: &mut [u8]) -> libc::c_long { -1 }

    fn getrandom_fill_bytes(v: &mut [u8]) {
        let mut read = 0;
        let len = v.len();
        while read < len {
            let result = getrandom(v[mut read..]);
            if result == -1 {
                let err = errno() as libc::c_int;
                if err == libc::EINTR {
                    continue;
                } else {
                    panic!("unexpected getrandom error: {}", err);
                }
            } else {
                read += result as uint;
            }
        }
    }

    fn getrandom_next_u32() -> u32 {
        let mut buf: [u8, ..4] = [0u8, ..4];
        getrandom_fill_bytes(&mut buf);
        unsafe { mem::transmute::<[u8, ..4], u32>(buf) }
    }

    fn getrandom_next_u64() -> u64 {
        let mut buf: [u8, ..8] = [0u8, ..8];
        getrandom_fill_bytes(&mut buf);
        unsafe { mem::transmute::<[u8, ..8], u64>(buf) }
    }

    #[cfg(all(target_os = "linux",
              any(target_arch = "x86_64", target_arch = "x86", target_arch = "arm")))]
    fn is_getrandom_available() -> bool {
        use sync::atomic::{AtomicBool, INIT_ATOMIC_BOOL, Relaxed};

        static GETRANDOM_CHECKED: AtomicBool = INIT_ATOMIC_BOOL;
        static GETRANDOM_AVAILABLE: AtomicBool = INIT_ATOMIC_BOOL;

        if !GETRANDOM_CHECKED.load(Relaxed) {
            let mut buf: [u8, ..0] = [];
            let result = getrandom(&mut buf);
            let available = if result == -1 {
                let err = errno() as libc::c_int;
                err != libc::ENOSYS
            } else {
                true
            };
            GETRANDOM_AVAILABLE.store(available, Relaxed);
            GETRANDOM_CHECKED.store(true, Relaxed);
            available
        } else {
            GETRANDOM_AVAILABLE.load(Relaxed)
        }
    }

    #[cfg(not(all(target_os = "linux",
                  any(target_arch = "x86_64", target_arch = "x86", target_arch = "arm"))))]
    fn is_getrandom_available() -> bool { false }

    /// A random number generator that retrieves randomness straight from
    /// the operating system. Platform sources:
    ///
    /// - Unix-like systems (Linux, Android, Mac OSX): read directly from
    ///   `/dev/urandom`, or from `getrandom(2)` system call if available.
    /// - Windows: calls `CryptGenRandom`, using the default cryptographic
    ///   service provider with the `PROV_RSA_FULL` type.
    /// - iOS: calls SecRandomCopyBytes as /dev/(u)random is sandboxed.
    ///
    /// This does not block.
    pub struct OsRng {
        inner: OsRngInner,
    }

    enum OsRngInner {
        OsGetrandomRng,
        OsReaderRng(ReaderRng<File>),
    }

    impl OsRng {
        /// Create a new `OsRng`.
        pub fn new() -> IoResult<OsRng> {
            if is_getrandom_available() {
                return Ok(OsRng { inner: OsGetrandomRng });
            }

            let reader = try!(File::open(&Path::new("/dev/urandom")));
            let reader_rng = ReaderRng::new(reader);

            Ok(OsRng { inner: OsReaderRng(reader_rng) })
        }
    }

    impl Rng for OsRng {
        fn next_u32(&mut self) -> u32 {
            match self.inner {
                OsGetrandomRng => getrandom_next_u32(),
                OsReaderRng(ref mut rng) => rng.next_u32(),
            }
        }
        fn next_u64(&mut self) -> u64 {
            match self.inner {
                OsGetrandomRng => getrandom_next_u64(),
                OsReaderRng(ref mut rng) => rng.next_u64(),
            }
        }
        fn fill_bytes(&mut self, v: &mut [u8]) {
            match self.inner {
                OsGetrandomRng => getrandom_fill_bytes(v),
                OsReaderRng(ref mut rng) => rng.fill_bytes(v)
            }
        }
    }
}

#[cfg(target_os = "ios")]
mod imp {
    extern crate libc;

    use io::{IoResult};
    use mem;
    use os;
    use rand::Rng;
    use result::Result::{Ok};
    use self::libc::{c_int, size_t};
    use slice::SliceExt;

    /// A random number generator that retrieves randomness straight from
    /// the operating system. Platform sources:
    ///
    /// - Unix-like systems (Linux, Android, Mac OSX): read directly from
    ///   `/dev/urandom`, or from `getrandom(2)` system call if available.
    /// - Windows: calls `CryptGenRandom`, using the default cryptographic
    ///   service provider with the `PROV_RSA_FULL` type.
    /// - iOS: calls SecRandomCopyBytes as /dev/(u)random is sandboxed.
    ///
    /// This does not block.
    #[allow(missing_copy_implementations)]
    pub struct OsRng {
        // dummy field to ensure that this struct cannot be constructed outside of this module
        _dummy: (),
    }

    #[repr(C)]
    struct SecRandom;

    #[allow(non_upper_case_globals)]
    static kSecRandomDefault: *const SecRandom = 0 as *const SecRandom;

    #[link(name = "Security", kind = "framework")]
    extern "C" {
        fn SecRandomCopyBytes(rnd: *const SecRandom,
                              count: size_t, bytes: *mut u8) -> c_int;
    }

    impl OsRng {
        /// Create a new `OsRng`.
        pub fn new() -> IoResult<OsRng> {
            Ok(OsRng { _dummy: () })
        }
    }

    impl Rng for OsRng {
        fn next_u32(&mut self) -> u32 {
            let mut v = [0u8, .. 4];
            self.fill_bytes(&mut v);
            unsafe { mem::transmute(v) }
        }
        fn next_u64(&mut self) -> u64 {
            let mut v = [0u8, .. 8];
            self.fill_bytes(&mut v);
            unsafe { mem::transmute(v) }
        }
        fn fill_bytes(&mut self, v: &mut [u8]) {
            let ret = unsafe {
                SecRandomCopyBytes(kSecRandomDefault, v.len() as size_t, v.as_mut_ptr())
            };
            if ret == -1 {
                panic!("couldn't generate random bytes: {}", os::last_os_error());
            }
        }
    }
}

#[cfg(windows)]
mod imp {
    extern crate libc;

    use io::{IoResult, IoError};
    use mem;
    use ops::Drop;
    use os;
    use rand::Rng;
    use result::Result::{Ok, Err};
    use self::libc::{DWORD, BYTE, LPCSTR, BOOL};
    use self::libc::types::os::arch::extra::{LONG_PTR};
    use slice::SliceExt;

    type HCRYPTPROV = LONG_PTR;

    /// A random number generator that retrieves randomness straight from
    /// the operating system. Platform sources:
    ///
    /// - Unix-like systems (Linux, Android, Mac OSX): read directly from
    ///   `/dev/urandom`, or from `getrandom(2)` system call if available.
    /// - Windows: calls `CryptGenRandom`, using the default cryptographic
    ///   service provider with the `PROV_RSA_FULL` type.
    /// - iOS: calls SecRandomCopyBytes as /dev/(u)random is sandboxed.
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
            self.fill_bytes(&mut v);
            unsafe { mem::transmute(v) }
        }
        fn next_u64(&mut self) -> u64 {
            let mut v = [0u8, .. 8];
            self.fill_bytes(&mut v);
            unsafe { mem::transmute(v) }
        }
        fn fill_bytes(&mut self, v: &mut [u8]) {
            let ret = unsafe {
                CryptGenRandom(self.hcryptprov, v.len() as DWORD,
                               v.as_mut_ptr())
            };
            if ret == 0 {
                panic!("couldn't generate random bytes: {}", os::last_os_error());
            }
        }
    }

    impl Drop for OsRng {
        fn drop(&mut self) {
            let ret = unsafe {
                CryptReleaseContext(self.hcryptprov, 0)
            };
            if ret == 0 {
                panic!("couldn't release context: {}", os::last_os_error());
            }
        }
    }
}

#[cfg(test)]
mod test {
    use prelude::*;

    use super::OsRng;
    use rand::Rng;
    use thread::Thread;

    #[test]
    fn test_os_rng() {
        let mut r = OsRng::new().unwrap();

        r.next_u32();
        r.next_u64();

        let mut v = [0u8, .. 1000];
        r.fill_bytes(&mut v);
    }

    #[test]
    fn test_os_rng_tasks() {

        let mut txs = vec!();
        for _ in range(0u, 20) {
            let (tx, rx) = channel();
            txs.push(tx);

            Thread::spawn(move|| {
                // wait until all the tasks are ready to go.
                rx.recv();

                // deschedule to attempt to interleave things as much
                // as possible (XXX: is this a good test?)
                let mut r = OsRng::new().unwrap();
                Thread::yield_now();
                let mut v = [0u8, .. 1000];

                for _ in range(0u, 100) {
                    r.next_u32();
                    Thread::yield_now();
                    r.next_u64();
                    Thread::yield_now();
                    r.fill_bytes(&mut v);
                    Thread::yield_now();
                }
            }).detach();
        }

        // start all the tasks
        for tx in txs.iter() {
            tx.send(())
        }
    }
}
