//! Store the ID of the main thread.
//!
//! The thread handle for the main thread is created lazily, and this might even
//! happen pre-main. Since not every platform has a way to identify the main
//! thread when that happens – macOS's `pthread_main_np` function being a notable
//! exception – we cannot assign it the right name right then. Instead, in our
//! runtime startup code, we remember the thread ID of the main thread (through
//! this modules `set` function) and use it to identify the main thread from then
//! on. This works reliably and has the additional advantage that we can report
//! the right thread name on main even after the thread handle has been destroyed.
//! Note however that this also means that the name reported in pre-main functions
//! will be incorrect, but that's just something we have to live with.

cfg_select! {
    target_has_atomic = "64" => {
        use super::id::ThreadId;
        use crate::sync::atomic::{Atomic, AtomicU64};
        use crate::sync::atomic::Ordering::Relaxed;

        static MAIN: Atomic<u64> = AtomicU64::new(0);

        pub(super) fn get() -> Option<ThreadId> {
            ThreadId::from_u64(MAIN.load(Relaxed))
        }

        /// # Safety
        /// May only be called once.
        pub(crate) unsafe fn set(id: ThreadId) {
            MAIN.store(id.as_u64().get(), Relaxed)
        }
    }
    _ => {
        use super::id::ThreadId;
        use crate::mem::MaybeUninit;
        use crate::sync::atomic::{Atomic, AtomicBool};
        use crate::sync::atomic::Ordering::{Acquire, Release};

        static INIT: Atomic<bool> = AtomicBool::new(false);
        static mut MAIN: MaybeUninit<ThreadId> = MaybeUninit::uninit();

        pub(super) fn get() -> Option<ThreadId> {
            if INIT.load(Acquire) {
                Some(unsafe { MAIN.assume_init() })
            } else {
                None
            }
        }

        /// # Safety
        /// May only be called once.
        pub(crate) unsafe fn set(id: ThreadId) {
            unsafe { MAIN = MaybeUninit::new(id) };
            INIT.store(true, Release);
        }
    }
}
