// Copyright 2013-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use mem;
use slice;

pub fn hashmap_random_keys() -> (u64, u64) {
    let mut v = (0, 0);
    unsafe {
        let view = slice::from_raw_parts_mut(&mut v as *mut _ as *mut u8,
                                             mem::size_of_val(&v));
        imp::fill_bytes(view);
    }
    return v
}

#[cfg(all(unix,
          not(target_os = "ios"),
          not(target_os = "openbsd"),
          not(target_os = "freebsd"),
          not(target_os = "fuchsia")))]
mod imp {
    use fs::File;
    use io::Read;
    use libc;
    use sys::os::errno;

    #[cfg(any(target_os = "linux", target_os = "android"))]
    fn getrandom(buf: &mut [u8]) -> libc::c_long {
        unsafe {
            libc::syscall(libc::SYS_getrandom, buf.as_mut_ptr(), buf.len(), libc::GRND_NONBLOCK)
        }
    }

    #[cfg(not(any(target_os = "linux", target_os = "android")))]
    fn getrandom(_buf: &mut [u8]) -> libc::c_long { -1 }

    fn getrandom_fill_bytes(v: &mut [u8]) -> bool {
        let mut read = 0;
        while read < v.len() {
            let result = getrandom(&mut v[read..]);
            if result == -1 {
                let err = errno() as libc::c_int;
                if err == libc::EINTR {
                    continue;
                } else if err == libc::EAGAIN {
                    return false
                } else {
                    panic!("unexpected getrandom error: {}", err);
                }
            } else {
                read += result as usize;
            }
        }

        return true
    }

    #[cfg(any(target_os = "linux", target_os = "android"))]
    fn is_getrandom_available() -> bool {
        use io;
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

    #[cfg(not(any(target_os = "linux", target_os = "android")))]
    fn is_getrandom_available() -> bool { false }

    pub fn fill_bytes(v: &mut [u8]) {
        // getrandom_fill_bytes here can fail if getrandom() returns EAGAIN,
        // meaning it would have blocked because the non-blocking pool (urandom)
        // has not initialized in the kernel yet due to a lack of entropy the
        // fallback we do here is to avoid blocking applications which could
        // depend on this call without ever knowing they do and don't have a
        // work around.  The PRNG of /dev/urandom will still be used but not
        // over a completely full entropy pool
        if is_getrandom_available() && getrandom_fill_bytes(v) {
            return
        }

        let mut file = File::open("/dev/urandom")
            .expect("failed to open /dev/urandom");
        file.read_exact(v).expect("failed to read /dev/urandom");
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
    use libc::{c_int, size_t};
    use ptr;

    enum SecRandom {}

    #[allow(non_upper_case_globals)]
    const kSecRandomDefault: *const SecRandom = ptr::null();

    extern {
        fn SecRandomCopyBytes(rnd: *const SecRandom,
                              count: size_t,
                              bytes: *mut u8) -> c_int;
    }

    pub fn fill_bytes(v: &mut [u8]) {
        let ret = unsafe {
            SecRandomCopyBytes(kSecRandomDefault,
                               v.len(),
                               v.as_mut_ptr())
        };
        if ret == -1 {
            panic!("couldn't generate random bytes: {}",
                   io::Error::last_os_error());
        }
    }
}

#[cfg(target_os = "freebsd")]
mod imp {
    use libc;
    use ptr;

    pub fn fill_bytes(v: &mut [u8]) {
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

#[cfg(target_os = "fuchsia")]
mod imp {
    #[link(name = "zircon")]
    extern {
        fn zx_cprng_draw(buffer: *mut u8, len: usize, actual: *mut usize) -> i32;
    }

    fn getrandom(buf: &mut [u8]) -> Result<usize, i32> {
        unsafe {
            let mut actual = 0;
            let status = zx_cprng_draw(buf.as_mut_ptr(), buf.len(), &mut actual);
            if status == 0 {
                Ok(actual)
            } else {
                Err(status)
            }
        }
    }

    pub fn fill_bytes(v: &mut [u8]) {
        let mut buf = v;
        while !buf.is_empty() {
            let ret = getrandom(buf);
            match ret {
                Err(err) => {
                    panic!("kernel zx_cprng_draw call failed! (returned {}, buf.len() {})",
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
