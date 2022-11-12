#![cfg(not(any(
    target_os = "linux",
    target_os = "android",
    all(target_os = "emscripten", target_feature = "atomics"),
    target_os = "freebsd",
    target_os = "openbsd",
    target_os = "dragonfly",
)))]

cfg_if::cfg_if! {
    if #[cfg(target_os = "fuchsia")] {
        mod fuchsia_mutex;
        pub(crate) use fuchsia_mutex::Mutex;
    } else {
        mod pthread_mutex;
        mod pthread_rwlock;
        mod pthread_condvar;
        pub(crate) use pthread_mutex::Mutex;
        pub(crate) use pthread_rwlock::RwLock;
        pub(crate) use pthread_condvar::Condvar;
    }
}
