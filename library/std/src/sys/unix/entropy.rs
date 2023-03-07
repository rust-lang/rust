use crate::io::{default_read, BorrowedCursor, Read, Result};

pub const INSECURE_HASHMAP: bool = false;

pub struct Entropy {
    pub insecure: bool,
}

cfg_if::cfg_if! {
    if #[cfg(any(
        target_os = "ios",
        target_os = "watchos",
        target_os = "openbsd",
        target_os = "freebsd",
        target_os = "netbsd",
        target_os = "fuchsia",
        target_os = "vxworks",
        target_os = "espidf",
        target_os = "horizon",
        target_os = "emscripten"
    ))] {
        impl Read for Entropy {
            fn read(&mut self, buf: &mut [u8]) -> Result<usize> {
                default_read(self, buf)
            }

            fn read_buf(&mut self, buf: BorrowedCursor<'_>) -> Result<()> {
                syscall(buf, self.insecure)
            }

            // On some platforms, the syscall always fills the complete buffer.
            #[cfg(any(
                target_os = "ios",
                target_os = "watchos",
                target_os = "fuchsia",
            ))]
            fn read_buf_exact(&mut self, buf: BorrowedCursor<'_>) -> Result<()> {
                syscall(buf, self.insecure)
            }
        }
    } else if #[cfg(any(
        target_os = "linux",
        target_os = "android",
        target_os = "macos",
    ))] {
        use crate::fs::File;

        impl Read for Entropy {
            fn read(&mut self, buf: &mut [u8]) -> Result<usize> {
                default_read(self, buf)
            }

            fn read_buf(&mut self, mut buf: BorrowedCursor<'_>) -> Result<()> {
                syscall(buf.reborrow(), self.insecure).unwrap_or_else(|| {
                    let mut file = file(self.insecure)?;
                    file.read_buf(buf)
                })
            }
        }
    } else {
        use crate::fs::File;

        impl Read for Entropy {
            fn read(&mut self, buf: &mut [u8]) -> Result<usize> {
                default_read(self, buf)
            }

            fn read_buf(&mut self, buf: BorrowedCursor<'_>) -> Result<()> {
                let mut file = file(self.insecure)?;
                file.read_buf(buf)
            }

            fn read_buf_exact(&mut self, buf: BorrowedCursor<'_>) -> Result<()> {
                let mut file = file(self.insecure)?;
                file.read_buf_exact(buf)
            }
        }
    }
}

#[cfg(any(target_os = "macos", target_os = "redox", target_os = "haiku"))]
fn file(_: bool) -> Result<&'static File> {
    use crate::sync::OnceLock;

    static URANDOM: OnceLock<File> = OnceLock::new();

    #[cfg(not(target_os = "redox"))]
    let path = "/dev/urandom";
    #[cfg(target_os = "redox")]
    let path = "rand:";
    URANDOM.get_or_try_init(|| File::open(path))
}

#[cfg(any(target_os = "linux", target_os = "android"))]
fn file(insecure: bool) -> Result<&'static File> {
    use crate::io::{Error, ErrorKind};
    use crate::os::fd::AsRawFd;
    use crate::sync::atomic::{AtomicBool, Ordering::Relaxed};
    use crate::sync::OnceLock;

    static URANDOM: OnceLock<File> = OnceLock::new();
    static IS_READY: AtomicBool = AtomicBool::new(false);

    let urandom = URANDOM.get_or_try_init(|| File::open("/dev/urandom"))?;

    while !insecure && !IS_READY.load(Relaxed) {
        let random = File::open("/dev/random")?;

        let mut pollfd = libc::pollfd { fd: random.as_raw_fd(), events: libc::POLLIN, revents: 0 };
        let res = unsafe { libc::poll(&mut pollfd, 1, -1) };
        match (res < 0).then(|| Error::last_os_error()) {
            None => {
                IS_READY.store(true, Relaxed);
                break;
            }
            Some(e) if e.kind() == ErrorKind::Interrupted => continue,
            Some(e) => return Err(e),
        }
    }

    Ok(urandom)
}

#[cfg(not(any(
    target_os = "linux",
    target_os = "android",
    target_os = "macos",
    target_os = "ios",
    target_os = "watchos",
    target_os = "openbsd",
    target_os = "freebsd",
    target_os = "netbsd",
    target_os = "fuchsia",
    target_os = "vxworks",
    target_os = "redox",
    target_os = "espidf",
    target_os = "horizon",
)))]
fn file(insecure: bool) -> Result<&'static File> {
    use crate::sync::OnceLock;

    static RANDOM: OnceLock<File> = OnceLock::new();
    static URANDOM: OnceLock<File> = OnceLock::new();

    if insecure {
        URANDOM.get_or_try_init(|| File::open("/dev/urandom"))
    } else {
        RANDOM.get_or_try_init(|| File::open("/dev/random"))
    }
}

#[cfg(any(target_os = "linux", target_os = "android"))]
fn syscall(mut buf: BorrowedCursor<'_>, insecure: bool) -> Option<Result<()>> {
    use crate::io::Error;
    use crate::sync::atomic::{AtomicBool, Ordering::Relaxed};
    use crate::sys::os::errno;
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

    static GRND_AVAILABLE: AtomicBool = AtomicBool::new(true);

    // This provides the best quality random numbers available at the given moment
    // without ever blocking, and is preferable to falling back to /dev/urandom.
    static GRND_INSECURE_AVAILABLE: AtomicBool = AtomicBool::new(true);

    if !GRND_AVAILABLE.load(Relaxed) {
        return None;
    }

    loop {
        let flags = if insecure && GRND_INSECURE_AVAILABLE.load(Relaxed) {
            libc::GRND_INSECURE
        } else if insecure {
            libc::GRND_NONBLOCK
        } else {
            0
        };

        let ret = unsafe { getrandom(buf.as_ptr().cast(), buf.capacity(), flags) };
        if ret != -1 {
            unsafe {
                buf.advance(ret as usize);
                return Some(Ok(()));
            }
        } else {
            match errno() {
                // Only returned if a flag is incorrect. This is only encountered on
                // systems where GRND_INSECURE is not available, so we remember that
                // and try again with GRND_NONBLOCK.
                libc::EINVAL => GRND_INSECURE_AVAILABLE.store(false, Relaxed),
                // Fall back to /dev/urandom to generate insecure data, as GRND_INSECURE
                // is not available.
                libc::EAGAIN => return None,
                // If getrandom is not available or blocked by seccomp, fall back to
                // the file method.
                libc::ENOSYS | libc::EPERM => {
                    GRND_AVAILABLE.store(false, Relaxed);
                    return None;
                }
                err => return Some(Err(Error::from_raw_os_error(err))),
            }
        }
    }
}

#[cfg(target_os = "macos")]
fn syscall(mut buf: BorrowedCursor<'_>, _: bool) -> Option<Result<()>> {
    use crate::io::Error;
    use crate::sys::weak::weak;
    use libc::{c_int, c_void, size_t};

    weak!(fn getentropy(*mut c_void, size_t) -> c_int);

    let getentropy = getentropy.get()?;
    // getentropy(2) permits a maximum buffer size of 256 bytes
    let len = usize::min(buf.capacity(), 256);
    let ret = unsafe { getentropy(buf.as_ptr().cast(), len) };
    if ret != -1 {
        unsafe {
            buf.advance(len);
            Some(Ok(()))
        }
    } else {
        Some(Err(Error::last_os_error()))
    }
}

#[cfg(any(target_os = "openbsd", target_os = "emscripten"))]
fn syscall(mut buf: BorrowedCursor<'_>, _: bool) -> Result<()> {
    use crate::io::Error;

    // getentropy(2) permits a maximum buffer size of 256 bytes
    let len = usize::min(buf.capacity(), 256);
    let ret = unsafe { libc::getentropy(buf.as_ptr().cast(), len) };
    if ret != -1 {
        unsafe {
            buf.advance(len);
            Ok(())
        }
    } else {
        Err(Error::last_os_error())
    }
}

// On iOS and MacOS `SecRandomCopyBytes` calls `CCRandomCopyBytes` with
// `kCCRandomDefault`. `CCRandomCopyBytes` manages a CSPRNG which is seeded
// from `/dev/random` and which runs on its own thread accessed via GCD.
// This seems needlessly heavyweight for the purposes of generating two u64s
// once per thread in `hashmap_random_keys`. Therefore `SecRandomCopyBytes` is
// only used on iOS where direct access to `/dev/urandom` is blocked by the
// sandbox.
#[cfg(any(target_os = "ios", target_os = "watchos"))]
fn syscall(mut buf: BorrowedCursor<'_>, _: bool) -> Result<()> {
    use crate::io::Error;
    use crate::ptr;
    use libc::{c_int, size_t};

    enum SecRandom {}

    #[allow(non_upper_case_globals)]
    const kSecRandomDefault: *const SecRandom = ptr::null();

    extern "C" {
        fn SecRandomCopyBytes(rnd: *const SecRandom, count: size_t, bytes: *mut u8) -> c_int;
    }

    let ret = unsafe { SecRandomCopyBytes(kSecRandomDefault, buf.capacity(), buf.as_ptr()) };
    if ret != -1 {
        unsafe {
            buf.advance(buf.capacity());
            Ok(())
        }
    } else {
        Err(Error::last_os_error())
    }
}

#[cfg(any(target_os = "freebsd", target_os = "netbsd"))]
fn syscall(mut buf: BorrowedCursor<'_>, _: bool) -> Result<()> {
    use crate::io::Error;
    use crate::ptr;

    let mib = [libc::CTL_KERN, libc::KERN_ARND];
    // kern.arandom permits a maximum buffer size of 256 bytes
    let mut len = usize::min(buf.capacity(), 256);
    let ret = unsafe {
        libc::sysctl(
            mib.as_ptr(),
            mib.len() as libc::c_uint,
            buf.as_ptr().cast(),
            &mut len,
            ptr::null(),
            0,
        )
    };

    if ret != -1 {
        unsafe {
            buf.advance(len);
            Ok(())
        }
    } else {
        Err(Error::last_os_error())
    }
}

#[cfg(target_os = "fuchsia")]
fn syscall(mut buf: BorrowedCursor<'_>, _: bool) -> Result<()> {
    #[link(name = "zircon")]
    extern "C" {
        fn zx_cprng_draw(buffer: *mut u8, len: usize);
    }

    unsafe {
        zx_cprng_draw(buf.as_ptr(), buf.capacity());
        buf.advance(buf.capacity());
        Ok(())
    }
}

#[cfg(target_os = "vxworks")]
fn syscall(mut buf: BorrowedCursor<'_>, _: bool) -> Result<()> {
    use crate::io::Error;
    use crate::sync::atomic::{AtomicBool, Ordering::Relaxed};

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

    let len = buf.capacity().try_into().unwrap_or(i32::MAX);
    let ret = unsafe { libc::randABytes(buf.as_ptr().cast(), len) };
    if ret == 0 {
        unsafe {
            buf.advance(len);
            Ok(())
        }
    } else {
        Err(Error::last_os_error())
    }
}

#[cfg(any(target_os = "espidf", target_os = "horizon"))]
fn syscall(mut buf: BorrowedCursor<'_>, _: bool) -> Result<()> {
    use crate::io::Error;

    let ret = unsafe { libc::getrandom(buf.as_ptr().cast(), buf.capacity(), 0) };
    if ret != -1 {
        unsafe {
            buf.advance(ret as usize);
            Ok(())
        }
    } else {
        Err(Error::last_os_error())
    }
}
