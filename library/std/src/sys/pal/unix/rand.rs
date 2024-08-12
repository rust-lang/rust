pub fn hashmap_random_keys() -> (u64, u64) {
    const KEY_LEN: usize = core::mem::size_of::<u64>();

    let mut v = [0u8; KEY_LEN * 2];
    if let Err(err) = read(&mut v) {
        panic!("failed to retrieve random hash map seed: {err}");
    }

    let key1 = v[0..KEY_LEN].try_into().unwrap();
    let key2 = v[KEY_LEN..].try_into().unwrap();

    (u64::from_ne_bytes(key1), u64::from_ne_bytes(key2))
}

cfg_if::cfg_if! {
    if #[cfg(any(
        target_vendor = "apple",
        target_os = "openbsd",
        target_os = "emscripten",
        target_os = "vita",
        all(target_os = "netbsd", not(netbsd10)),
        target_os = "fuchsia",
        target_os = "vxworks",
    ))] {
        // Some systems have a syscall that directly retrieves random data.
        // If that is guaranteed to be available, use it.
        use imp::syscall as read;
    } else {
        // Otherwise, try the syscall to see if it exists only on some systems
        // and fall back to reading from the random device otherwise.
        fn read(bytes: &mut [u8]) -> crate::io::Result<()> {
            use crate::fs::File;
            use crate::io::Read;
            use crate::sync::OnceLock;

            #[cfg(any(
                target_os = "linux",
                target_os = "android",
                target_os = "espidf",
                target_os = "horizon",
                target_os = "freebsd",
                target_os = "dragonfly",
                target_os = "solaris",
                target_os = "illumos",
                netbsd10,
            ))]
            if let Some(res) = imp::syscall(bytes) {
                return res;
            }

            const PATH: &'static str = if cfg!(target_os = "redox") {
                "/scheme/rand"
            } else {
                "/dev/urandom"
            };

            static FILE: OnceLock<File> = OnceLock::new();

            FILE.get_or_try_init(|| File::open(PATH))?.read_exact(bytes)
        }
    }
}

// All these systems a `getrandom` syscall.
//
// It is not guaranteed to be available, so return None to fallback to the file
// implementation.
#[cfg(any(
    target_os = "linux",
    target_os = "android",
    target_os = "espidf",
    target_os = "horizon",
    target_os = "freebsd",
    target_os = "dragonfly",
    target_os = "solaris",
    target_os = "illumos",
    netbsd10,
))]
mod imp {
    use crate::io::{Error, Result};
    use crate::sync::atomic::{AtomicBool, Ordering};
    use crate::sys::os::errno;

    #[cfg(any(target_os = "linux", target_os = "android"))]
    fn getrandom(buf: &mut [u8]) -> libc::ssize_t {
        use crate::sys::weak::syscall;

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
        target_os = "dragonfly",
        target_os = "espidf",
        target_os = "horizon",
        target_os = "freebsd",
        netbsd10,
        target_os = "illumos",
        target_os = "solaris"
    ))]
    fn getrandom(buf: &mut [u8]) -> libc::ssize_t {
        unsafe { libc::getrandom(buf.as_mut_ptr().cast(), buf.len(), 0) }
    }

    pub fn syscall(v: &mut [u8]) -> Option<Result<()>> {
        static GETRANDOM_UNAVAILABLE: AtomicBool = AtomicBool::new(false);

        if GETRANDOM_UNAVAILABLE.load(Ordering::Relaxed) {
            return None;
        }

        let mut read = 0;
        while read < v.len() {
            let result = getrandom(&mut v[read..]);
            if result == -1 {
                let err = errno() as libc::c_int;
                if err == libc::EINTR {
                    continue;
                } else if err == libc::ENOSYS || err == libc::EPERM {
                    // `getrandom` is not supported on the current system.
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
                    return None;
                } else if err == libc::EAGAIN {
                    // getrandom has failed because it would have blocked as the
                    // non-blocking pool (urandom) has not been initialized in
                    // the kernel yet due to a lack of entropy. Fallback to
                    // reading from `/dev/urandom` which will return potentially
                    // insecure random data to avoid blocking applications which
                    // could depend on this call without ever knowing they do and
                    // don't have a work around.
                    return None;
                } else {
                    return Some(Err(Error::from_raw_os_error(err)));
                }
            } else {
                read += result as usize;
            }
        }

        Some(Ok(()))
    }
}

#[cfg(any(
    target_os = "macos", // Supported since macOS 10.12+.
    target_os = "openbsd",
    target_os = "emscripten",
    target_os = "vita",
))]
mod imp {
    use crate::io::{Error, Result};

    pub fn syscall(v: &mut [u8]) -> Result<()> {
        // getentropy(2) permits a maximum buffer size of 256 bytes
        for s in v.chunks_mut(256) {
            let ret = unsafe { libc::getentropy(s.as_mut_ptr().cast(), s.len()) };
            if ret == -1 {
                return Err(Error::last_os_error());
            }
        }

        Ok(())
    }
}

// On Apple platforms, `CCRandomGenerateBytes` and `SecRandomCopyBytes` simply
// call into `CCRandomCopyBytes` with `kCCRandomDefault`. `CCRandomCopyBytes`
// manages a CSPRNG which is seeded from the kernel's CSPRNG and which runs on
// its own thread accessed via GCD. This seems needlessly heavyweight for our purposes
// so we only use it when `getentropy` is blocked, which appears to be the case
// on all platforms except macOS (see #102643).
//
// `CCRandomGenerateBytes` is used instead of `SecRandomCopyBytes` because the former is accessible
// via `libSystem` (libc) while the other needs to link to `Security.framework`.
#[cfg(all(target_vendor = "apple", not(target_os = "macos")))]
mod imp {
    use libc::size_t;

    use crate::ffi::{c_int, c_void};
    use crate::io::{Error, Result};

    pub fn syscall(v: &mut [u8]) -> Result<()> {
        extern "C" {
            fn CCRandomGenerateBytes(bytes: *mut c_void, count: size_t) -> c_int;
        }

        let ret = unsafe { CCRandomGenerateBytes(v.as_mut_ptr().cast(), v.len()) };
        if ret != -1 { Ok(()) } else { Err(Error::last_os_error()) }
    }
}

// FIXME: once the 10.x release becomes the minimum, this can be dropped for simplification.
#[cfg(all(target_os = "netbsd", not(netbsd10)))]
mod imp {
    use crate::io::{Error, Result};
    use crate::ptr;

    pub fn syscall(v: &mut [u8]) -> Result<()> {
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
            if ret == -1 {
                return Err(Error::last_os_error());
            } else if s_len != s.len() {
                // FIXME(joboet): this can't actually happen, can it?
                panic!("read less bytes than requested from kern.arandom");
            }
        }

        Ok(())
    }
}

#[cfg(target_os = "fuchsia")]
mod imp {
    use crate::io::Result;

    #[link(name = "zircon")]
    extern "C" {
        fn zx_cprng_draw(buffer: *mut u8, len: usize);
    }

    pub fn syscall(v: &mut [u8]) -> Result<()> {
        unsafe { zx_cprng_draw(v.as_mut_ptr(), v.len()) };
        Ok(())
    }
}

#[cfg(target_os = "vxworks")]
mod imp {
    use core::sync::atomic::AtomicBool;
    use core::sync::atomic::Ordering::Relaxed;

    use crate::io::{Error, Result};

    pub fn syscall(v: &mut [u8]) -> Result<()> {
        static RNG_INIT: AtomicBool = AtomicBool::new(false);
        while !RNG_INIT.load(Relaxed) {
            let ret = unsafe { libc::randSecure() };
            if ret < 0 {
                return Err(Error::last_os_error());
            } else if ret > 0 {
                RNG_INIT.store(true, Relaxed);
                break;
            }

            unsafe { libc::usleep(10) };
        }

        let ret = unsafe {
            libc::randABytes(v.as_mut_ptr() as *mut libc::c_uchar, v.len() as libc::c_int)
        };
        if ret >= 0 { Ok(()) } else { Err(Error::last_os_error()) }
    }
}
