// Copyright 2013-2015 The Rust Project Developers. See the COPYRIGHT
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

#[cfg(all(unix, not(target_os = "ios"), not(target_os = "openbsd")))]
mod imp {
    use self::OsRngInner::*;

    use fs::File;
    use io;
    use libc;
    use mem;
    use rand::Rng;
    use rand::reader::ReaderRng;
    use sys::os::errno;

    #[cfg(all(target_os = "linux",
              any(target_arch = "x86_64",
                  target_arch = "x86",
                  target_arch = "arm",
                  target_arch = "aarch64",
                  target_arch = "powerpc")))]
    fn getrandom(buf: &mut [u8]) -> libc::c_long {
        extern "C" {
            fn syscall(number: libc::c_long, ...) -> libc::c_long;
        }

        #[cfg(target_arch = "x86_64")]
        const NR_GETRANDOM: libc::c_long = 318;
        #[cfg(target_arch = "x86")]
        const NR_GETRANDOM: libc::c_long = 355;
        #[cfg(any(target_arch = "arm", target_arch = "powerpc"))]
        const NR_GETRANDOM: libc::c_long = 384;
        #[cfg(target_arch = "aarch64")]
        const NR_GETRANDOM: libc::c_long = 278;

        unsafe {
            syscall(NR_GETRANDOM, buf.as_mut_ptr(), buf.len(), 0)
        }
    }

    #[cfg(not(all(target_os = "linux",
                  any(target_arch = "x86_64",
                      target_arch = "x86",
                      target_arch = "arm",
                      target_arch = "aarch64",
                      target_arch = "powerpc"))))]
    fn getrandom(_buf: &mut [u8]) -> libc::c_long { -1 }

    fn getrandom_fill_bytes(v: &mut [u8]) {
        let mut read = 0;
        while read < v.len() {
            let result = getrandom(&mut v[read..]);
            if result == -1 {
                let err = errno() as libc::c_int;
                if err == libc::EINTR {
                    continue;
                } else {
                    panic!("unexpected getrandom error: {}", err);
                }
            } else {
                read += result as usize;
            }
        }
    }

    fn getrandom_next_u32() -> u32 {
        let mut buf: [u8; 4] = [0; 4];
        getrandom_fill_bytes(&mut buf);
        unsafe { mem::transmute::<[u8; 4], u32>(buf) }
    }

    fn getrandom_next_u64() -> u64 {
        let mut buf: [u8; 8] = [0; 8];
        getrandom_fill_bytes(&mut buf);
        unsafe { mem::transmute::<[u8; 8], u64>(buf) }
    }

    #[cfg(all(target_os = "linux",
              any(target_arch = "x86_64",
                  target_arch = "x86",
                  target_arch = "arm",
                  target_arch = "aarch64",
                  target_arch = "powerpc")))]
    fn is_getrandom_available() -> bool {
        use sync::atomic::{AtomicBool, Ordering};
        use sync::Once;

        static CHECKER: Once = Once::new();
        static AVAILABLE: AtomicBool = AtomicBool::new(false);

        CHECKER.call_once(|| {
            let mut buf: [u8; 0] = [];
            let result = getrandom(&mut buf);
            let available = if result == -1 {
                let err = io::Error::last_os_error().raw_os_error();
                err != Some(libc::ENOSYS)
            } else {
                true
            };
            AVAILABLE.store(available, Ordering::Relaxed);
        });

        AVAILABLE.load(Ordering::Relaxed)
    }

    #[cfg(not(all(target_os = "linux",
                  any(target_arch = "x86_64",
                      target_arch = "x86",
                      target_arch = "arm",
                      target_arch = "aarch64",
                      target_arch = "powerpc"))))]
    fn is_getrandom_available() -> bool { false }

    /// A random number generator that retrieves randomness straight from
    /// the operating system. Platform sources:
    ///
    /// - Unix-like systems (Linux, Android, Mac OSX): read directly from
    ///   `/dev/urandom`, or from `getrandom(2)` system call if available.
    /// - Windows: calls `CryptGenRandom`, using the default cryptographic
    ///   service provider with the `PROV_RSA_FULL` type.
    /// - iOS: calls SecRandomCopyBytes as /dev/(u)random is sandboxed.
    /// - OpenBSD: uses the `getentropy(2)` system call.
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
        pub fn new() -> io::Result<OsRng> {
            if is_getrandom_available() {
                return Ok(OsRng { inner: OsGetrandomRng });
            }

            let reader = try!(File::open("/dev/urandom"));
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

#[cfg(target_os = "openbsd")]
mod imp {
    use io;
    use libc;
    use mem;
    use sys::os::errno;
    use rand::Rng;

    /// A random number generator that retrieves randomness straight from
    /// the operating system. Platform sources:
    ///
    /// - Unix-like systems (Linux, Android, Mac OSX): read directly from
    ///   `/dev/urandom`, or from `getrandom(2)` system call if available.
    /// - Windows: calls `CryptGenRandom`, using the default cryptographic
    ///   service provider with the `PROV_RSA_FULL` type.
    /// - iOS: calls SecRandomCopyBytes as /dev/(u)random is sandboxed.
    /// - OpenBSD: uses the `getentropy(2)` system call.
    ///
    /// This does not block.
    pub struct OsRng {
        // dummy field to ensure that this struct cannot be constructed outside
        // of this module
        _dummy: (),
    }

    impl OsRng {
        /// Create a new `OsRng`.
        pub fn new() -> io::Result<OsRng> {
            Ok(OsRng { _dummy: () })
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
            // getentropy(2) permits a maximum buffer size of 256 bytes
            for s in v.chunks_mut(256) {
                let ret = unsafe {
                    libc::syscall(libc::NR_GETENTROPY, s.as_mut_ptr(), s.len())
                };
                if ret == -1 {
                    panic!("unexpected getentropy error: {}", errno());
                }
            }
        }
    }
}

#[cfg(target_os = "ios")]
mod imp {
    use io;
    use mem;
    use ptr;
    use rand::Rng;
    use libc::{c_int, size_t};

    /// A random number generator that retrieves randomness straight from
    /// the operating system. Platform sources:
    ///
    /// - Unix-like systems (Linux, Android, Mac OSX): read directly from
    ///   `/dev/urandom`, or from `getrandom(2)` system call if available.
    /// - Windows: calls `CryptGenRandom`, using the default cryptographic
    ///   service provider with the `PROV_RSA_FULL` type.
    /// - iOS: calls SecRandomCopyBytes as /dev/(u)random is sandboxed.
    /// - OpenBSD: uses the `getentropy(2)` system call.
    ///
    /// This does not block.
    pub struct OsRng {
        // dummy field to ensure that this struct cannot be constructed outside
        // of this module
        _dummy: (),
    }

    enum SecRandom {}

    #[allow(non_upper_case_globals)]
    const kSecRandomDefault: *const SecRandom = ptr::null();

    #[link(name = "Security", kind = "framework")]
    extern "C" {
        fn SecRandomCopyBytes(rnd: *const SecRandom,
                              count: size_t, bytes: *mut u8) -> c_int;
    }

    impl OsRng {
        /// Create a new `OsRng`.
        pub fn new() -> io::Result<OsRng> {
            Ok(OsRng { _dummy: () })
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
            let ret = unsafe {
                SecRandomCopyBytes(kSecRandomDefault, v.len() as size_t,
                                   v.as_mut_ptr())
            };
            if ret == -1 {
                panic!("couldn't generate random bytes: {}",
                       io::Error::last_os_error());
            }
        }
    }
}

#[cfg(windows)]
mod imp {
    use io;
    use mem;
    use rand::Rng;
    use sys::c;

    /// A random number generator that retrieves randomness straight from
    /// the operating system. Platform sources:
    ///
    /// - Unix-like systems (Linux, Android, Mac OSX): read directly from
    ///   `/dev/urandom`, or from `getrandom(2)` system call if available.
    /// - Windows: calls `CryptGenRandom`, using the default cryptographic
    ///   service provider with the `PROV_RSA_FULL` type.
    /// - iOS: calls SecRandomCopyBytes as /dev/(u)random is sandboxed.
    /// - OpenBSD: uses the `getentropy(2)` system call.
    ///
    /// This does not block.
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
            let ret = unsafe {
                c::CryptGenRandom(self.hcryptprov, v.len() as c::DWORD,
                                  v.as_mut_ptr())
            };
            if ret == 0 {
                panic!("couldn't generate random bytes: {}",
                       io::Error::last_os_error());
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
}

#[cfg(test)]
mod tests {
    use prelude::v1::*;

    use sync::mpsc::channel;
    use rand::Rng;
    use super::OsRng;
    use thread;

    #[test]
    fn test_os_rng() {
        let mut r = OsRng::new().unwrap();

        r.next_u32();
        r.next_u64();

        let mut v = [0; 1000];
        r.fill_bytes(&mut v);
    }

    #[test]
    fn test_os_rng_tasks() {

        let mut txs = vec!();
        for _ in 0..20 {
            let (tx, rx) = channel();
            txs.push(tx);

            thread::spawn(move|| {
                // wait until all the threads are ready to go.
                rx.recv().unwrap();

                // deschedule to attempt to interleave things as much
                // as possible (XXX: is this a good test?)
                let mut r = OsRng::new().unwrap();
                thread::yield_now();
                let mut v = [0; 1000];

                for _ in 0..100 {
                    r.next_u32();
                    thread::yield_now();
                    r.next_u64();
                    thread::yield_now();
                    r.fill_bytes(&mut v);
                    thread::yield_now();
                }
            });
        }

        // start all the threads
        for tx in &txs {
            tx.send(()).unwrap();
        }
    }
}
