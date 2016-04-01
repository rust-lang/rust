// Copyright 2013-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Interface to the operating system provided random number
//! generators:
//!
//! - Unix-like systems (Linux, Android, Mac OSX): read directly from
//!   `/dev/urandom`, or from `getrandom(2)` system call if available.
//! - Windows: calls `CryptGenRandom`, using the default cryptographic
//!   service provider with the `PROV_RSA_FULL` type.
//! - iOS: calls SecRandomCopyBytes as /dev/(u)random is sandboxed.
//! - OpenBSD: uses the `getentropy(2)` system call.


pub use self::imp::fill_bytes;

#[cfg(all(unix, not(target_os = "ios"), not(target_os = "openbsd")))]
mod imp {
    use fs::File;
    use io::{self, Read};
    use libc;
    use sys::os::errno;

    #[cfg(all(target_os = "linux",
              any(target_arch = "x86_64",
                  target_arch = "x86",
                  target_arch = "arm",
                  target_arch = "aarch64",
                  target_arch = "powerpc",
                  target_arch = "powerpc64",
                  target_arch = "powerpc64le")))]
    fn getrandom(buf: &mut [u8]) -> libc::c_long {
        #[cfg(target_arch = "x86_64")]
        const NR_GETRANDOM: libc::c_long = 318;
        #[cfg(target_arch = "x86")]
        const NR_GETRANDOM: libc::c_long = 355;
        #[cfg(target_arch = "arm")]
        const NR_GETRANDOM: libc::c_long = 384;
        #[cfg(any(target_arch = "powerpc", target_arch = "powerpc64",
                  target_arch = "powerpc64le"))]
        const NR_GETRANDOM: libc::c_long = 359;
        #[cfg(target_arch = "aarch64")]
        const NR_GETRANDOM: libc::c_long = 278;

        unsafe {
            libc::syscall(NR_GETRANDOM, buf.as_mut_ptr(), buf.len(), 0)
        }
    }

    #[cfg(not(all(target_os = "linux",
                  any(target_arch = "x86_64",
                      target_arch = "x86",
                      target_arch = "arm",
                      target_arch = "aarch64",
                      target_arch = "powerpc",
                      target_arch = "powerpc64",
                      target_arch = "powerpc64le"))))]
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

    #[cfg(all(target_os = "linux",
              any(target_arch = "x86_64",
                  target_arch = "x86",
                  target_arch = "arm",
                  target_arch = "aarch64",
                  target_arch = "powerpc",
                  target_arch = "powerpc64",
                  target_arch = "powerpc64le")))]
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
                      target_arch = "powerpc",
                      target_arch = "powerpc64",
                      target_arch = "powerpc64le"))))]
    fn is_getrandom_available() -> bool { false }

    pub fn fill_bytes(b: &mut [u8]) {
        if is_getrandom_available() {
            getrandom_fill_bytes(b)
        } else {
            let mut reader = File::open("/dev/urandom").expect("failed to open /dev/urandom");
            reader.read_exact(b).expect("failed to read bytes from /dev/urandom");
        }
    }
}

#[cfg(target_os = "openbsd")]
mod imp {
    use libc;
    use sys::os::errno;

    pub fn fill_bytes(v: &mut [u8]) {
        // getentropy(2) permits a maximum buffer size of 256 bytes
        for s in v.chunks_mut(256) {
            let ret = unsafe {
                libc::getentropy(s.as_mut_ptr() as *mut libc::c_void, s.len())
            };
            if ret == -1 {
                panic!("unexpected getentropy error: {}", errno());
            }
        }
    }
}

#[cfg(target_os = "ios")]
mod imp {
    use io;
    use ptr;
    use libc::{c_int, size_t};

    enum SecRandom {}

    #[allow(non_upper_case_globals)]
    const kSecRandomDefault: *const SecRandom = ptr::null();

    #[link(name = "Security", kind = "framework")]
    extern "C" {
        fn SecRandomCopyBytes(rnd: *const SecRandom,
                              count: size_t, bytes: *mut u8) -> c_int;
    }

    pub fn fill_bytes(v: &mut [u8]) {
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

#[cfg(windows)]
mod imp {
    use io;
    use sys::c;

    // SBRM struct to ensure the cryptography context gets cleaned up
    // properly
    struct OsRng {
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

    pub fn fill_bytes(v: &mut [u8]) {
        let os = OsRng::new().expect("failed to acquire crypt context for randomness");
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

#[cfg(test)]
mod tests {
    use sync::mpsc::channel;
    use super::fill_bytes;
    use thread;

    #[test]
    fn test_fill_bytes() {
        let mut v = [0; 1000];
        fill_bytes(&mut v);
    }

    #[test]
    fn test_fill_bytes_tasks() {

        let mut txs = vec!();
        for _ in 0..20 {
            let (tx, rx) = channel();
            txs.push(tx);

            thread::spawn(move|| {
                // wait until all the threads are ready to go.
                rx.recv().unwrap();

                let mut v = [0; 1000];

                for _ in 0..100 {
                    fill_bytes(&mut v);
                    // deschedule to attempt to interleave things as much
                    // as possible (XXX: is this a good test?)
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
