cfg_if::cfg_if! {
    if #[cfg(any(
        target_os = "linux",
        target_os = "android",
    ))] {
        mod futex;
        mod futex_rwlock;
        pub use futex::{Mutex, MovableMutex, Condvar, MovableCondvar, ReentrantMutex};
        pub use futex_rwlock::{RwLock, MovableRwLock};
    } else {
        mod pthread_mutex;
        mod pthread_remutex;
        mod pthread_rwlock;
        mod pthread_condvar;
        pub use pthread_mutex::{Mutex, MovableMutex};
        pub use pthread_remutex::ReentrantMutex;
        pub use pthread_rwlock::{RwLock, MovableRwLock};
        pub use pthread_condvar::{Condvar, MovableCondvar};
    }
}
