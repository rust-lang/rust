cfg_if::cfg_if! {
    if #[cfg(any(
        target_os = "linux",
        target_os = "android",
        all(target_os = "emscripten", target_feature = "atomics"),
        target_os = "freebsd",
        target_os = "openbsd",
        target_os = "dragonfly",
    ))] {
        mod futex_mutex;
        mod futex_rwlock;
        mod futex_condvar;
        pub(crate) use futex_mutex::{Mutex, MovableMutex};
        pub(crate) use futex_rwlock::MovableRwLock;
        pub(crate) use futex_condvar::MovableCondvar;
    } else if #[cfg(target_os = "fuchsia")] {
        mod fuchsia_mutex;
        mod futex_rwlock;
        mod futex_condvar;
        pub(crate) use fuchsia_mutex::{Mutex, MovableMutex};
        pub(crate) use futex_rwlock::MovableRwLock;
        pub(crate) use futex_condvar::MovableCondvar;
    } else {
        mod pthread_mutex;
        mod pthread_rwlock;
        mod pthread_condvar;
        pub(crate) use pthread_mutex::{Mutex, MovableMutex};
        pub(crate) use pthread_rwlock::MovableRwLock;
        pub(crate) use pthread_condvar::MovableCondvar;
    }
}
