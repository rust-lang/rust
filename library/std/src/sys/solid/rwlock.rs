//! A readers-writer lock implementation backed by the SOLID kernel extension.
use super::{
    abi,
    itron::{
        error::{expect_success, expect_success_aborting, fail, ItronError},
        spin::SpinIdOnceCell,
    },
};

pub struct RWLock {
    /// The ID of the underlying mutex object
    rwl: SpinIdOnceCell<()>,
}

pub type MovableRWLock = RWLock;

// Safety: `num_readers` is protected by `mtx_num_readers`
unsafe impl Send for RWLock {}
unsafe impl Sync for RWLock {}

fn new_rwl() -> Result<abi::ID, ItronError> {
    ItronError::err_if_negative(unsafe { abi::rwl_acre_rwl() })
}

impl RWLock {
    pub const fn new() -> RWLock {
        RWLock { rwl: SpinIdOnceCell::new() }
    }

    /// Get the inner mutex's ID, which is lazily created.
    fn raw(&self) -> abi::ID {
        match self.rwl.get_or_try_init(|| new_rwl().map(|id| (id, ()))) {
            Ok((id, ())) => id,
            Err(e) => fail(e, &"rwl_acre_rwl"),
        }
    }

    #[inline]
    pub unsafe fn read(&self) {
        let rwl = self.raw();
        expect_success(unsafe { abi::rwl_loc_rdl(rwl) }, &"rwl_loc_rdl");
    }

    #[inline]
    pub unsafe fn try_read(&self) -> bool {
        let rwl = self.raw();
        match unsafe { abi::rwl_ploc_rdl(rwl) } {
            abi::E_TMOUT => false,
            er => {
                expect_success(er, &"rwl_ploc_rdl");
                true
            }
        }
    }

    #[inline]
    pub unsafe fn write(&self) {
        let rwl = self.raw();
        expect_success(unsafe { abi::rwl_loc_wrl(rwl) }, &"rwl_loc_wrl");
    }

    #[inline]
    pub unsafe fn try_write(&self) -> bool {
        let rwl = self.raw();
        match unsafe { abi::rwl_ploc_wrl(rwl) } {
            abi::E_TMOUT => false,
            er => {
                expect_success(er, &"rwl_ploc_wrl");
                true
            }
        }
    }

    #[inline]
    pub unsafe fn read_unlock(&self) {
        let rwl = self.raw();
        expect_success_aborting(unsafe { abi::rwl_unl_rwl(rwl) }, &"rwl_unl_rwl");
    }

    #[inline]
    pub unsafe fn write_unlock(&self) {
        let rwl = self.raw();
        expect_success_aborting(unsafe { abi::rwl_unl_rwl(rwl) }, &"rwl_unl_rwl");
    }

    #[inline]
    pub unsafe fn destroy(&self) {
        if let Some(rwl) = self.rwl.get().map(|x| x.0) {
            expect_success_aborting(unsafe { abi::rwl_del_rwl(rwl) }, &"rwl_del_rwl");
        }
    }
}
