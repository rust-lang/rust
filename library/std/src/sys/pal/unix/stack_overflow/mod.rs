#![cfg_attr(test, allow(dead_code))]
#![forbid(unsafe_op_in_unsafe_fn)]

pub use self::imp::init;
use self::imp::{drop_handler, make_handler};

pub struct Handler {
    data: *mut libc::c_void,
}

impl Handler {
    pub unsafe fn new() -> Handler {
        make_handler(false)
    }

    fn null() -> Handler {
        Handler { data: crate::ptr::null_mut() }
    }
}

impl Drop for Handler {
    fn drop(&mut self) {
        unsafe {
            drop_handler(self.data);
        }
    }
}

#[cfg(all(
    not(miri),
    any(
        target_os = "linux",
        target_os = "freebsd",
        target_os = "hurd",
        target_os = "macos",
        target_os = "netbsd",
        target_os = "openbsd",
        target_os = "solaris",
        target_os = "illumos",
    ),
))]
mod thread_info;

// miri doesn't model signals nor stack overflows and this code has some
// synchronization properties that we don't want to expose to user code,
// hence we disable it on miri.
#[cfg(all(
    not(miri),
    any(
        target_os = "linux",
        target_os = "freebsd",
        target_os = "hurd",
        target_os = "macos",
        target_os = "netbsd",
        target_os = "openbsd",
        target_os = "solaris",
        target_os = "illumos",
    )
))]
mod imp;

// This is intentionally not enabled on iOS/tvOS/watchOS/visionOS, as it uses
// several symbols that might lead to rejections from the App Store, namely
// `sigaction`, `sigaltstack`, `sysctlbyname`, `mmap`, `munmap` and `mprotect`.
//
// This might be overly cautious, though it is also what Swift does (and they
// usually have fewer qualms about forwards compatibility, since the runtime
// is shipped with the OS):
// <https://github.com/apple/swift/blob/swift-5.10-RELEASE/stdlib/public/runtime/CrashHandlerMacOS.cpp>
#[cfg(any(
    miri,
    not(any(
        target_os = "linux",
        target_os = "freebsd",
        target_os = "hurd",
        target_os = "macos",
        target_os = "netbsd",
        target_os = "openbsd",
        target_os = "solaris",
        target_os = "illumos",
        target_os = "cygwin",
    ))
))]
mod imp;

#[cfg(target_os = "cygwin")]
mod imp;
