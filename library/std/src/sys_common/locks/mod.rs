cfg_if::cfg_if! {
    if #[cfg(any(
        target_os = "android",
        target_os = "dragonfly",
        all(target_os = "emscripten", target_feature = "atomics"),
        target_os = "freebsd",
        target_os = "hermit",
        target_os = "linux",
        target_os = "openbsd",
        all(any(target_arch = "wasm32", target_arch = "wasm64"), target_feature = "atomics"),
    ))] {
        mod futex_mutex;
        mod futex_rwlock;
        mod futex_condvar;
        pub(crate) use futex_mutex::Mutex;
        pub(crate) use futex_rwlock::RwLock;
        pub(crate) use futex_condvar::Condvar;
    } else if #[cfg(target_os = "fuchsia")] {
        mod futex_rwlock;
        mod futex_condvar;
        pub(crate) use crate::sys::locks::Mutex;
        pub(crate) use futex_rwlock::RwLock;
        pub(crate) use futex_condvar::Condvar;
    } else {
        pub(crate) use crate::sys::locks::{Mutex, RwLock, Condvar};
    }
}
