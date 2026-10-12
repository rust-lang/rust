use crate::cell::Cell;

#[cfg(all(
    target_has_threads,
    // VEXos is only "target_has_threads" because it allows user interrupt handlers which preempt
    // the main thread, but they are forbidden from accessing sync primitives (the same as UNIX
    // signal handlers).
    not(target_os = "vexos"),
))]
compile_error!("Using no_threads implementation on a target with threads");

pub struct Mutex {
    // This platform has no threads, so we can use a Cell here.
    locked: Cell<bool>,
}

unsafe impl Send for Mutex {}
unsafe impl Sync for Mutex {} // no threads on this platform

impl Mutex {
    #[inline]
    pub const fn new() -> Mutex {
        Mutex { locked: Cell::new(false) }
    }

    #[inline]
    pub fn lock(&self) {
        assert_eq!(self.locked.replace(true), false, "cannot recursively acquire mutex");
    }

    #[inline]
    pub unsafe fn unlock(&self) {
        self.locked.set(false);
    }

    #[inline]
    pub fn try_lock(&self) -> bool {
        !self.locked.replace(true)
    }
}
