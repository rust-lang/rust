// Copyright 2013-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

pub use self::imp::OsRng;

use mem;

fn next_u32(mut fill_buf: &mut FnMut(&mut [u8])) -> u32 {
    let mut buf: [u8; 4] = [0; 4];
    fill_buf(&mut buf);
    unsafe { mem::transmute::<[u8; 4], u32>(buf) }
}

fn next_u64(mut fill_buf: &mut FnMut(&mut [u8])) -> u64 {
    let mut buf: [u8; 8] = [0; 8];
    fill_buf(&mut buf);
    unsafe { mem::transmute::<[u8; 8], u64>(buf) }
}

#[cfg(all(unix,
          not(target_os = "ios"),
          not(target_os = "openbsd"),
          not(target_os = "freebsd"),
          not(target_os = "fuchsia")))]
mod imp {
    use self::OsRngInner::*;
    use super::{next_u32, next_u64};

    use fs::File;
    use io;
    use libc;
    use rand::Rng;
    use rand::reader::ReaderRng;
    use sys::os::errno;

    #[cfg(all(target_os = "linux",
              any(target_arch = "x86_64",
                  target_arch = "x86",
                  target_arch = "arm",
                  target_arch = "aarch64",
                  target_arch = "powerpc",
                  target_arch = "powerpc64",
                  target_arch = "s390x")))]
    fn getrandom(buf: &mut [u8]) -> libc::c_long {
        #[cfg(target_arch = "x86_64")]
        const NR_GETRANDOM: libc::c_long = 318;
        #[cfg(target_arch = "x86")]
        const NR_GETRANDOM: libc::c_long = 355;
        #[cfg(target_arch = "arm")]
        const NR_GETRANDOM: libc::c_long = 384;
        #[cfg(target_arch = "s390x")]
        const NR_GETRANDOM: libc::c_long = 349;
        #[cfg(any(target_arch = "powerpc", target_arch = "powerpc64"))]
        const NR_GETRANDOM: libc::c_long = 359;
        #[cfg(target_arch = "aarch64")]
        const NR_GETRANDOM: libc::c_long = 278;

        const GRND_NONBLOCK: libc::c_uint = 0x0001;

        unsafe {
            libc::syscall(NR_GETRANDOM, buf.as_mut_ptr(), buf.len(), GRND_NONBLOCK)
        }
    }

    #[cfg(not(all(target_os = "linux",
                  any(target_arch = "x86_64",
                      target_arch = "x86",
                      target_arch = "arm",
                      target_arch = "aarch64",
                      target_arch = "powerpc",
                      target_arch = "powerpc64",
                      target_arch = "s390x"))))]
    fn getrandom(_buf: &mut [u8]) -> libc::c_long { -1 }

    fn getrandom_fill_bytes(v: &mut [u8]) {
        let mut read = 0;
        while read < v.len() {
            let result = getrandom(&mut v[read..]);
            if result == -1 {
                let err = errno() as libc::c_int;
                if err == libc::EINTR {
                    continue;
                } else if err == libc::EAGAIN {
                    // if getrandom() returns EAGAIN it would have blocked
                    // because the non-blocking pool (urandom) has not
                    // initialized in the kernel yet due to a lack of entropy
                    // the fallback we do here is to avoid blocking applications
                    // which could depend on this call without ever knowing
                    // they do and don't have a work around. The PRNG of
                    // /dev/urandom will still be used but not over a completely
                    // full entropy pool
                    let reader = File::open("/dev/urandom").expect("Unable to open /dev/urandom");
                    let mut reader_rng = ReaderRng::new(reader);
                    reader_rng.fill_bytes(&mut v[read..]);
                    read += v.len();
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
                  target_arch = "s390x")))]
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
                      target_arch = "s390x"))))]
    fn is_getrandom_available() -> bool { false }

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

            let reader = File::open("/dev/urandom")?;
            let reader_rng = ReaderRng::new(reader);

            Ok(OsRng { inner: OsReaderRng(reader_rng) })
        }
    }

    impl Rng for OsRng {
        fn next_u32(&mut self) -> u32 {
            match self.inner {
                OsGetrandomRng => next_u32(&mut getrandom_fill_bytes),
                OsReaderRng(ref mut rng) => rng.next_u32(),
            }
        }
        fn next_u64(&mut self) -> u64 {
            match self.inner {
                OsGetrandomRng => next_u64(&mut getrandom_fill_bytes),
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
    use super::{next_u32, next_u64};

    use io;
    use libc;
    use sys::os::errno;
    use rand::Rng;

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
            next_u32(&mut |v| self.fill_bytes(v))
        }
        fn next_u64(&mut self) -> u64 {
            next_u64(&mut |v| self.fill_bytes(v))
        }
        fn fill_bytes(&mut self, v: &mut [u8]) {
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
}

#[cfg(target_os = "ios")]
mod imp {
    use super::{next_u32, next_u64};

    use io;
    use ptr;
    use rand::Rng;
    use libc::{c_int, size_t};

    pub struct OsRng {
        // dummy field to ensure that this struct cannot be constructed outside
        // of this module
        _dummy: (),
    }

    enum SecRandom {}

    #[allow(non_upper_case_globals)]
    const kSecRandomDefault: *const SecRandom = ptr::null();

    #[link(name = "Security", kind = "framework")]
    #[cfg(not(cargobuild))]
    extern {}

    extern {
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
            next_u32(&mut |v| self.fill_bytes(v))
        }
        fn next_u64(&mut self) -> u64 {
            next_u64(&mut |v| self.fill_bytes(v))
        }
        fn fill_bytes(&mut self, v: &mut [u8]) {
            let ret = unsafe {
                SecRandomCopyBytes(kSecRandomDefault, v.len(),
                                   v.as_mut_ptr())
            };
            if ret == -1 {
                panic!("couldn't generate random bytes: {}",
                       io::Error::last_os_error());
            }
        }
    }
}

#[cfg(target_os = "freebsd")]
mod imp {
    use super::{next_u32, next_u64};

    use io;
    use libc;
    use rand::Rng;
    use ptr;

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
            next_u32(&mut |v| self.fill_bytes(v))
        }
        fn next_u64(&mut self) -> u64 {
            next_u64(&mut |v| self.fill_bytes(v))
        }
        fn fill_bytes(&mut self, v: &mut [u8]) {
            let mib = [libc::CTL_KERN, libc::KERN_ARND];
            // kern.arandom permits a maximum buffer size of 256 bytes
            for s in v.chunks_mut(256) {
                let mut s_len = s.len();
                let ret = unsafe {
                    libc::sysctl(mib.as_ptr(), mib.len() as libc::c_uint,
                                 s.as_mut_ptr() as *mut _, &mut s_len,
                                 ptr::null(), 0)
                };
                if ret == -1 || s_len != s.len() {
                    panic!("kern.arandom sysctl failed! (returned {}, s.len() {}, oldlenp {})",
                           ret, s.len(), s_len);
                }
            }
        }
    }
}

#[cfg(target_os = "fuchsia")]
mod imp {
    use super::{next_u32, next_u64};

    use io;
    use rand::Rng;

    #[link(name = "magenta")]
    extern {
        fn mx_cprng_draw(buffer: *mut u8, len: usize, actual: *mut usize) -> i32;
    }

    fn getrandom(buf: &mut [u8]) -> Result<usize, i32> {
        unsafe {
            let mut actual = 0;
            let status = mx_cprng_draw(buf.as_mut_ptr(), buf.len(), &mut actual);
            if status == 0 {
                Ok(actual)
            } else {
                Err(status)
            }
        }
    }

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
            next_u32(&mut |v| self.fill_bytes(v))
        }
        fn next_u64(&mut self) -> u64 {
            next_u64(&mut |v| self.fill_bytes(v))
        }
        fn fill_bytes(&mut self, v: &mut [u8]) {
            let mut buf = v;
            while !buf.is_empty() {
                let ret = getrandom(buf);
                match ret {
                    Err(err) => {
                        panic!("kernel mx_cprng_draw call failed! (returned {}, buf.len() {})",
                            err, buf.len())
                    }
                    Ok(actual) => {
                        let move_buf = buf;
                        buf = &mut move_buf[(actual as usize)..];
                    }
                }
            }
        }
    }
}
