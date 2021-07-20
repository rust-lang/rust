//! Mutex implementation backed by μITRON mutexes. Assumes `acre_mtx` and
//! `TA_INHERIT` are available.
use super::{
    abi,
    error::{expect_success, expect_success_aborting, fail, ItronError},
    spin::SpinIdOnceCell,
};
use crate::cell::UnsafeCell;

pub struct Mutex {
    /// The ID of the underlying mutex object
    mtx: SpinIdOnceCell<()>,
}

pub type MovableMutex = Mutex;

/// Create a mutex object. This function never panics.
fn new_mtx() -> Result<abi::ID, ItronError> {
    ItronError::err_if_negative(unsafe {
        abi::acre_mtx(&abi::T_CMTX {
            // Priority inheritance mutex
            mtxatr: abi::TA_INHERIT,
            // Unused
            ceilpri: 0,
        })
    })
}

impl Mutex {
    pub const fn new() -> Mutex {
        Mutex { mtx: SpinIdOnceCell::new() }
    }

    pub unsafe fn init(&mut self) {
        // Initialize `self.mtx` eagerly
        let id = new_mtx().unwrap_or_else(|e| fail(e, &"acre_mtx"));
        unsafe { self.mtx.set_unchecked((id, ())) };
    }

    /// Get the inner mutex's ID, which is lazily created.
    fn raw(&self) -> abi::ID {
        match self.mtx.get_or_try_init(|| new_mtx().map(|id| (id, ()))) {
            Ok((id, ())) => id,
            Err(e) => fail(e, &"acre_mtx"),
        }
    }

    pub unsafe fn lock(&self) {
        let mtx = self.raw();
        expect_success(unsafe { abi::loc_mtx(mtx) }, &"loc_mtx");
    }

    pub unsafe fn unlock(&self) {
        let mtx = unsafe { self.mtx.get_unchecked().0 };
        expect_success_aborting(unsafe { abi::unl_mtx(mtx) }, &"unl_mtx");
    }

    pub unsafe fn try_lock(&self) -> bool {
        let mtx = self.raw();
        match unsafe { abi::ploc_mtx(mtx) } {
            abi::E_TMOUT => false,
            er => {
                expect_success(er, &"ploc_mtx");
                true
            }
        }
    }

    pub unsafe fn destroy(&self) {
        if let Some(mtx) = self.mtx.get().map(|x| x.0) {
            expect_success_aborting(unsafe { abi::del_mtx(mtx) }, &"del_mtx");
        }
    }
}

pub(super) struct MutexGuard<'a>(&'a Mutex);

impl<'a> MutexGuard<'a> {
    #[inline]
    pub(super) fn lock(x: &'a Mutex) -> Self {
        unsafe { x.lock() };
        Self(x)
    }
}

impl Drop for MutexGuard<'_> {
    #[inline]
    fn drop(&mut self) {
        unsafe { self.0.unlock() };
    }
}

// All empty stubs because this platform does not yet support threads, so lock
// acquisition always succeeds.
pub struct ReentrantMutex {
    /// The ID of the underlying mutex object
    mtx: abi::ID,
    /// The lock count.
    count: UnsafeCell<usize>,
}

unsafe impl Send for ReentrantMutex {}
unsafe impl Sync for ReentrantMutex {}

impl ReentrantMutex {
    pub const unsafe fn uninitialized() -> ReentrantMutex {
        ReentrantMutex { mtx: 0, count: UnsafeCell::new(0) }
    }

    pub unsafe fn init(&mut self) {
        self.mtx = expect_success(
            unsafe {
                abi::acre_mtx(&abi::T_CMTX {
                    // Priority inheritance mutex
                    mtxatr: abi::TA_INHERIT,
                    // Unused
                    ceilpri: 0,
                })
            },
            &"acre_mtx",
        );
    }

    pub unsafe fn lock(&self) {
        match unsafe { abi::loc_mtx(self.mtx) } {
            abi::E_OBJ => {
                // Recursive lock
                unsafe {
                    let count = &mut *self.count.get();
                    if let Some(new_count) = count.checked_add(1) {
                        *count = new_count;
                    } else {
                        // counter overflow
                        rtabort!("lock count overflow");
                    }
                }
            }
            er => {
                expect_success(er, &"loc_mtx");
            }
        }
    }

    pub unsafe fn unlock(&self) {
        unsafe {
            let count = &mut *self.count.get();
            if *count > 0 {
                *count -= 1;
                return;
            }
        }

        expect_success_aborting(unsafe { abi::unl_mtx(self.mtx) }, &"unl_mtx");
    }

    pub unsafe fn try_lock(&self) -> bool {
        let er = unsafe { abi::ploc_mtx(self.mtx) };
        if er == abi::E_OBJ {
            // Recursive lock
            unsafe {
                let count = &mut *self.count.get();
                if let Some(new_count) = count.checked_add(1) {
                    *count = new_count;
                } else {
                    // counter overflow
                    rtabort!("lock count overflow");
                }
            }
            true
        } else if er == abi::E_TMOUT {
            // Locked by another thread
            false
        } else {
            expect_success(er, &"ploc_mtx");
            // Top-level lock by the current thread
            true
        }
    }

    pub unsafe fn destroy(&self) {
        expect_success_aborting(unsafe { abi::del_mtx(self.mtx) }, &"del_mtx");
    }
}
