pub use self::imp::OsRng as Rng;

#[cfg(all(unix, not(target_os = "ios")))]
mod imp {
    use self::OsRngInner::*;

    use os_str::prelude::*;
    use error::prelude::*;
    use io::prelude::*;
    use fs::prelude::*;
    use unix::cvt_r;
    use rand as sys;
    use libc;
    use core::mem;
    use core_rand as rand;

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
        #[cfg(any(target_arch = "aarch64"))]
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
        let len = v.len();
        while read < len {
            let result = match cvt_r(|| getrandom(&mut v[read..])) {
                Err(err) => panic!("unexpected getrandom error: {}", err),
                Ok(res) => res,
            };

            read += result as usize;
        }
    }

    #[cfg(all(target_os = "linux",
              any(target_arch = "x86_64",
                  target_arch = "x86",
                  target_arch = "arm",
                  target_arch = "aarch64",
                  target_arch = "powerpc")))]
    fn is_getrandom_available() -> bool {
        use core::sync::atomic::{AtomicBool, Ordering};
        use sync::prelude::*;

        static CHECKER: Once = Once::new();
        static AVAILABLE: AtomicBool = AtomicBool::new(false);

        CHECKER.call_once(|| {
            let mut buf: [u8; 0] = [];
            let result = getrandom(&mut buf);
            let available = if result == -1 {
                let err = Error::expect_last_error().code();
                err != libc::ENOSYS
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
    ///
    /// This does not block.
    pub struct OsRng {
        inner: OsRngInner,
    }

    enum OsRngInner {
        OsGetrandomRng,
        OsReaderRng(File),
    }

    impl sys::Rng for OsRng {
        /// Create a new `OsRng`.
        fn new() -> Result<OsRng> {
            if is_getrandom_available() {
                return Ok(OsRng { inner: OsGetrandomRng });
            }

            let mut opt = OpenOptions::new();
            opt.read(true);
            let reader = try!(File::open(OsStr::from_str("/dev/urandom"), &opt));

            Ok(OsRng { inner: OsReaderRng(reader) })
        }
    }

    impl rand::Rng for OsRng {
        fn next_u32(&mut self) -> u32 {
            let mut bytes = [0; 4];
            self.fill_bytes(&mut bytes);
            unsafe { mem::transmute(bytes) }
        }

        fn next_u64(&mut self) -> u64 {
            let mut bytes = [0; 8];
            self.fill_bytes(&mut bytes);
            unsafe { mem::transmute(bytes) }
        }

        fn fill_bytes(&mut self, mut v: &mut [u8]) {
            match self.inner {
                OsGetrandomRng => getrandom_fill_bytes(v),
                OsReaderRng(ref rng) => {
                    while !v.is_empty() {
                        let t = v;
                        match rng.read(t) {
                            Ok(0) => panic!("OsRng::fill_bytes: EOF reached"),
                            Ok(n) => v = t.split_at_mut(n).1,
                            Err(e) => panic!("OsRng::fill_bytes: {}", e),
                        }
                    }
                }
            }
        }
    }
}

#[cfg(target_os = "ios")]
mod imp {
    #[cfg(stage0)] use prelude::v1::*;

    use io;
    use mem;
    use ptr;
    use sys::rand as sys;
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

    impl sys::Rng for OsRng {
        /// Create a new `OsRng`.
        fn new() -> Result<OsRng> {
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
                       Error::expect_last_error());
            }
        }
    }
}
