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

mod condvar;
mod mutex;

pub use condvar::Condvar;
pub use mutex::Mutex;
