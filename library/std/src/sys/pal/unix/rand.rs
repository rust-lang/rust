pub fn hashmap_random_keys() -> (u64, u64) {
    const KEY_LEN: usize = core::mem::size_of::<u64>();

    let mut v = [0u8; KEY_LEN * 2];
    imp::fill_bytes(&mut v);

    let key1 = v[0..KEY_LEN].try_into().unwrap();
    let key2 = v[KEY_LEN..].try_into().unwrap();

    (u64::from_ne_bytes(key1), u64::from_ne_bytes(key2))
}

#[cfg(all(
    unix,
    not(target_os = "macos"),
    not(target_os = "ios"),
    not(target_os = "tvos"),
    not(target_os = "watchos"),
    not(target_os = "openbsd"),
    not(target_os = "netbsd"),
    not(target_os = "fuchsia"),
    not(target_os = "redox"),
    not(target_os = "vxworks"),
    not(target_os = "emscripten"),
    not(target_os = "vita"),
))]
mod imp {
    use crate::fs::File;
    use crate::io::Read;

    #[cfg(any(target_os = "linux", target_os = "android"))]
    use crate::sys::weak::syscall;

    #[cfg(any(target_os = "linux", target_os = "android"))]
    fn getrandom(buf: &mut [u8]) -> libc::ssize_t {
        use crate::sync::atomic::{AtomicBool, Ordering};
        use crate::sys::os::errno;

        // A weak symbol allows interposition, e.g. for perf measurements that want to
        // disable randomness for consistency. Otherwise, we'll try a raw syscall.
        // (`getrandom` was added in glibc 2.25, musl 1.1.20, android API level 28)
        syscall! {
            fn getrandom(
                buffer: *mut libc::c_void,
                length: libc::size_t,
                flags: libc::c_uint
            ) -> libc::ssize_t
        }

        // This provides the best quality random numbers available at the given moment
        // without ever blocking, and is preferable to falling back to /dev/urandom.
        static GRND_INSECURE_AVAILABLE: AtomicBool = AtomicBool::new(true);
        if GRND_INSECURE_AVAILABLE.load(Ordering::Relaxed) {
            let ret = unsafe { getrandom(buf.as_mut_ptr().cast(), buf.len(), libc::GRND_INSECURE) };
            if ret == -1 && errno() as libc::c_int == libc::EINVAL {
                GRND_INSECURE_AVAILABLE.store(false, Ordering::Relaxed);
            } else {
                return ret;
            }
        }

        unsafe { getrandom(buf.as_mut_ptr().cast(), buf.len(), libc::GRND_NONBLOCK) }
    }

    #[cfg(any(
        target_os = "espidf",
        target_os = "horizon",
        target_os = "freebsd",
        target_os = "dragonfly",
        netbsd10
    ))]
    fn getrandom(buf: &mut [u8]) -> libc::ssize_t {
        unsafe { libc::getrandom(buf.as_mut_ptr().cast(), buf.len(), 0) }
    }

    #[cfg(not(any(
        target_os = "linux",
        target_os = "android",
        target_os = "espidf",
        target_os = "horizon",
        target_os = "freebsd",
        target_os = "dragonfly",
        netbsd10
    )))]
    fn getrandom_fill_bytes(_buf: &mut [u8]) -> bool {
        false
    }

    #[cfg(any(
        target_os = "linux",
        target_os = "android",
        target_os = "espidf",
        target_os = "horizon",
        target_os = "freebsd",
        target_os = "dragonfly",
        netbsd10
    ))]
    fn getrandom_fill_bytes(v: &mut [u8]) -> bool {
        use crate::sync::atomic::{AtomicBool, Ordering};
        use crate::sys::os::errno;

        static GETRANDOM_UNAVAILABLE: AtomicBool = AtomicBool::new(false);
        if GETRANDOM_UNAVAILABLE.load(Ordering::Relaxed) {
            return false;
        }

        let mut read = 0;
        while read < v.len() {
            let result = getrandom(&mut v[read..]);
            if result == -1 {
                let err = errno() as libc::c_int;
                if err == libc::EINTR {
                    continue;
                } else if err == libc::ENOSYS || err == libc::EPERM {
                    // Fall back to reading /dev/urandom if `getrandom` is not
                    // supported on the current kernel.
                    //
                    // Also fall back in case it is disabled by something like
                    // seccomp or inside of docker.
                    //
                    // If the `getrandom` syscall is not implemented in the current kernel version it should return an
                    // `ENOSYS` error. Docker also blocks the whole syscall inside unprivileged containers, and
                    // returns `EPERM` (instead of `ENOSYS`) when a program tries to invoke the syscall. Because of
                    // that we need to check for *both* `ENOSYS` and `EPERM`.
                    //
                    // Note that Docker's behavior is breaking other projects (notably glibc), so they're planning
                    // to update their filtering to return `ENOSYS` in a future release:
                    //
                    //     https://github.com/moby/moby/issues/42680
                    //
                    GETRANDOM_UNAVAILABLE.store(true, Ordering::Relaxed);
                    return false;
                } else if err == libc::EAGAIN {
                    return false;
                } else {
                    panic!("unexpected getrandom error: {err}");
                }
            } else {
                read += result as usize;
            }
        }
        true
    }

    pub fn fill_bytes(v: &mut [u8]) {
        // getrandom_fill_bytes here can fail if getrandom() returns EAGAIN,
        // meaning it would have blocked because the non-blocking pool (urandom)
        // has not initialized in the kernel yet due to a lack of entropy. The
        // fallback we do here is to avoid blocking applications which could
        // depend on this call without ever knowing they do and don't have a
        // work around. The PRNG of /dev/urandom will still be used but over a
        // possibly predictable entropy pool.
        if getrandom_fill_bytes(v) {
            return;
        }

        // getrandom failed because it is permanently or temporarily (because
        // of missing entropy) unavailable. Open /dev/urandom, read from it,
        // and close it again.
        let mut file = File::open("/dev/urandom").expect("failed to open /dev/urandom");
        file.read_exact(v).expect("failed to read /dev/urandom")
    }
}

#[cfg(target_vendor = "apple")]
mod imp {
    use crate::io;
    use libc::{c_int, c_void, size_t};

    #[inline(always)]
    fn random_failure() -> ! {
        panic!("unexpected random generation error: {}", io::Error::last_os_error());
    }

    #[cfg(target_os = "macos")]
    fn getentropy_fill_bytes(v: &mut [u8]) {
        extern "C" {
            fn getentropy(bytes: *mut c_void, count: size_t) -> c_int;
        }

        // getentropy(2) permits a maximum buffer size of 256 bytes
        for s in v.chunks_mut(256) {
            let ret = unsafe { getentropy(s.as_mut_ptr().cast(), s.len()) };
            if ret == -1 {
                random_failure()
            }
        }
    }

    #[cfg(not(target_os = "macos"))]
    fn ccrandom_fill_bytes(v: &mut [u8]) {
        extern "C" {
            fn CCRandomGenerateBytes(bytes: *mut c_void, count: size_t) -> c_int;
        }

        let ret = unsafe { CCRandomGenerateBytes(v.as_mut_ptr().cast(), v.len()) };
        if ret == -1 {
            random_failure()
        }
    }

    pub fn fill_bytes(v: &mut [u8]) {
        // All supported versions of macOS (10.12+) support getentropy.
        //
        // `getentropy` is measurably faster (via Divan) then the other alternatives so its preferred
        // when usable.
        #[cfg(target_os = "macos")]
        getentropy_fill_bytes(v);

        // On Apple platforms, `CCRandomGenerateBytes` and `SecRandomCopyBytes` simply
        // call into `CCRandomCopyBytes` with `kCCRandomDefault`. `CCRandomCopyBytes`
        // manages a CSPRNG which is seeded from the kernel's CSPRNG and which runs on
        // its own thread accessed via GCD. This seems needlessly heavyweight for our purposes
        // so we only use it on non-Mac OSes where the better entrypoints are blocked.
        //
        // `CCRandomGenerateBytes` is used instead of `SecRandomCopyBytes` because the former is accessible
        // via `libSystem` (libc) while the other needs to link to `Security.framework`.
        //
        // Note that while `getentropy` has a available attribute in the macOS headers, the lack
        // of a header in the iOS (and others) SDK means that its can cause app store rejections.
        // Just use `CCRandomGenerateBytes` instead.
        #[cfg(not(target_os = "macos"))]
        ccrandom_fill_bytes(v);
    }
}

#[cfg(any(target_os = "openbsd", target_os = "emscripten", target_os = "vita"))]
mod imp {
    use crate::sys::os::errno;

    pub fn fill_bytes(v: &mut [u8]) {
        // getentropy(2) permits a maximum buffer size of 256 bytes
        for s in v.chunks_mut(256) {
            let ret = unsafe { libc::getentropy(s.as_mut_ptr() as *mut libc::c_void, s.len()) };
            if ret == -1 {
                panic!("unexpected getentropy error: {}", errno());
            }
        }
    }
}

// FIXME: once the 10.x release becomes the minimum, this can be dropped for simplification.
#[cfg(all(target_os = "netbsd", not(netbsd10)))]
mod imp {
    use crate::ptr;

    pub fn fill_bytes(v: &mut [u8]) {
        let mib = [libc::CTL_KERN, libc::KERN_ARND];
        // kern.arandom permits a maximum buffer size of 256 bytes
        for s in v.chunks_mut(256) {
            let mut s_len = s.len();
            let ret = unsafe {
                libc::sysctl(
                    mib.as_ptr(),
                    mib.len() as libc::c_uint,
                    s.as_mut_ptr() as *mut _,
                    &mut s_len,
                    ptr::null(),
                    0,
                )
            };
            if ret == -1 || s_len != s.len() {
                panic!(
                    "kern.arandom sysctl failed! (returned {}, s.len() {}, oldlenp {})",
                    ret,
                    s.len(),
                    s_len
                );
            }
        }
    }
}

#[cfg(target_os = "fuchsia")]
mod imp {
    #[link(name = "zircon")]
    extern "C" {
        fn zx_cprng_draw(buffer: *mut u8, len: usize);
    }

    pub fn fill_bytes(v: &mut [u8]) {
        unsafe { zx_cprng_draw(v.as_mut_ptr(), v.len()) }
    }
}

#[cfg(target_os = "redox")]
mod imp {
    use crate::fs::File;
    use crate::io::Read;

    pub fn fill_bytes(v: &mut [u8]) {
        // Open rand:, read from it, and close it again.
        let mut file = File::open("rand:").expect("failed to open rand:");
        file.read_exact(v).expect("failed to read rand:")
    }
}

#[cfg(target_os = "vxworks")]
mod imp {
    use crate::io;
    use core::sync::atomic::{AtomicBool, Ordering::Relaxed};

    pub fn fill_bytes(v: &mut [u8]) {
        static RNG_INIT: AtomicBool = AtomicBool::new(false);
        while !RNG_INIT.load(Relaxed) {
            let ret = unsafe { libc::randSecure() };
            if ret < 0 {
                panic!("couldn't generate random bytes: {}", io::Error::last_os_error());
            } else if ret > 0 {
                RNG_INIT.store(true, Relaxed);
                break;
            }
            unsafe { libc::usleep(10) };
        }
        let ret = unsafe {
            libc::randABytes(v.as_mut_ptr() as *mut libc::c_uchar, v.len() as libc::c_int)
        };
        if ret < 0 {
            panic!("couldn't generate random bytes: {}", io::Error::last_os_error());
        }
    }
}
