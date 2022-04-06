cfg_if::cfg_if! {
    if #[cfg(any(
        target_os = "linux",
        target_os = "android",
    ))] {
        mod futex;
        #[allow(dead_code)]
        mod pthread_mutex; // Only used for PthreadMutexAttr, needed by pthread_remutex.
        mod pthread_remutex; // FIXME: Implement this using a futex
        mod pthread_rwlock; // FIXME: Implement this using a futex
        pub use futex::{Mutex, MovableMutex, Condvar, MovableCondvar};
        pub use pthread_remutex::ReentrantMutex;
        pub use pthread_rwlock::{RwLock, MovableRwLock};
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
