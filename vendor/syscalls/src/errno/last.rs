#[cfg(any(
    target_os = "linux",
    target_os = "redox",
    target_os = "dragonfly",
    target_os = "fuchsia"
))]
mod ffi {
    extern "C" {
        pub fn __errno_location() -> *mut i32;
    }

    pub unsafe fn errno() -> *mut i32 {
        __errno_location()
    }
}

#[cfg(any(target_os = "android", target_os = "netbsd", target_os = "openbsd"))]
mod ffi {
    extern "C" {
        pub fn __errno() -> *mut i32;
    }

    pub unsafe fn errno() -> *mut i32 {
        __errno()
    }
}

#[cfg(any(target_os = "freebsd", target_os = "ios", target_os = "macos"))]
mod ffi {
    extern "C" {
        pub fn __error() -> *mut i32;
    }

    pub unsafe fn errno() -> *mut i32 {
        __error()
    }
}

#[cfg(any(target_os = "illumos", target_os = "solaris"))]
mod ffi {
    extern "C" {
        pub fn ___errno() -> *mut i32;
    }

    pub unsafe fn errno() -> *mut i32 {
        __errno()
    }
}

pub use ffi::errno;
