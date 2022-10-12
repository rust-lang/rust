//! Thread parking on systems without futex support.

#![cfg(not(any(
    target_os = "linux",
    target_os = "android",
    all(target_os = "emscripten", target_feature = "atomics"),
    target_os = "freebsd",
    target_os = "openbsd",
    target_os = "dragonfly",
    target_os = "fuchsia",
)))]

cfg_if::cfg_if! {
    if #[cfg(target_os = "netbsd")] {
        mod netbsd;
        pub use netbsd::Parker;
    } else {
        mod pthread;
        pub use pthread::Parker;
    }
}
