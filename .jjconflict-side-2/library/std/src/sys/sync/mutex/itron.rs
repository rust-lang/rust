//! Mutex implementation backed by μITRON mutexes. Assumes `acre_mtx` and
//! `TA_INHERIT` are available.
#![forbid(unsafe_op_in_unsafe_fn)]

use crate::sys::pal::itron::abi;
use crate::sys::pal::itron::error::{ItronError, expect_success, expect_success_aborting, fail};
use crate::sys::pal::itron::spin::SpinIdOnceCell;

pub struct Mutex {
    /// The ID of the underlying mutex object
    mtx: SpinIdOnceCell<()>,
}

/// Creates a mutex object. This function never panics.
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
    #[inline]
    pub const fn new() -> Mutex {
        Mutex { mtx: SpinIdOnceCell::new() }
    }

    /// Gets the inner mutex's ID, which is lazily created.
    fn raw(&self) -> abi::ID {
        match self.mtx.get_or_try_init(|| new_mtx().map(|id| (id, ()))) {
            Ok((id, ())) => id,
            Err(e) => fail(e, &"acre_mtx"),
        }
    }

    pub fn lock(&self) {
        let mtx = self.raw();
        expect_success(unsafe { abi::loc_mtx(mtx) }, &"loc_mtx");
    }

    pub unsafe fn unlock(&self) {
        let mtx = unsafe { self.mtx.get_unchecked().0 };
        expect_success_aborting(unsafe { abi::unl_mtx(mtx) }, &"unl_mtx");
    }

    pub fn try_lock(&self) -> bool {
        let mtx = self.raw();
        match unsafe { abi::ploc_mtx(mtx) } {
            abi::E_TMOUT => false,
            er => {
                expect_success(er, &"ploc_mtx");
                true
            }
        }
    }
}

impl Drop for Mutex {
    fn drop(&mut self) {
        if let Some(mtx) = self.mtx.get().map(|x| x.0) {
            expect_success_aborting(unsafe { abi::del_mtx(mtx) }, &"del_mtx");
        }
    }
}
