#![cfg(not(any(
    target_os = "linux",
    target_os = "android",
    all(target_os = "emscripten", target_feature = "atomics"),
    target_os = "freebsd",
    target_os = "openbsd",
    target_os = "dragonfly",
    target_os = "fuchsia",
)))]
#![forbid(unsafe_op_in_unsafe_fn)]

#[cfg(not(target_vendor = "apple"))]
mod condvar;
mod mutex;

#[cfg(not(target_vendor = "apple"))]
pub use condvar::Condvar;
pub use mutex::Mutex;
