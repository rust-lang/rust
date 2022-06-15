//! UEFI-specific extensions to the primitives in `std::env` module

use crate::ffi::c_void;
use crate::sync::atomic::{AtomicPtr, Ordering};
use r_efi::efi::{Handle, SystemTable};

static mut GLOBAL_SYSTEM_TABLE: GlobalData<SystemTable> = GlobalData::new();
static mut GLOBAL_SYSTEM_HANDLE: GlobalData<c_void> = GlobalData::new();

pub(crate) unsafe fn init_globals(
    handle: Handle,
    system_table: *mut SystemTable,
) -> Result<(), ()> {
    GLOBAL_SYSTEM_TABLE.init(system_table).map_err(|_| ())?;
    GLOBAL_SYSTEM_HANDLE.init(handle).map_err(|_| ())?;
    Ok(())
}

#[unstable(feature = "uefi_std", issue = "none")]
/// This function returns error if SystemTable pointer is null
pub unsafe fn get_system_table() -> Result<*mut SystemTable, ()> {
    GLOBAL_SYSTEM_TABLE.load()
}

#[unstable(feature = "uefi_std", issue = "none")]
/// This function returns error if SystemHandle pointer is null
pub unsafe fn get_system_handle() -> Result<Handle, ()> {
    GLOBAL_SYSTEM_HANDLE.load()
}

/// It is mostly ment to
/// store SystemTable and SystemHandle.
struct GlobalData<T> {
    ptr: AtomicPtr<T>,
}

impl<T> GlobalData<T> {
    /// Initializes GlobalData with internal NULL pointer. This is constant so that it can be used
    /// in statics.
    const fn new() -> Self {
        Self { ptr: AtomicPtr::new(core::ptr::null_mut()) }
    }

    /// SAFETY: This function will only initialize once.
    /// The return value is a Result containing nothing if it is success. In the case of an
    /// error, it returns the previous pointer.
    fn init(&self, ptr: *mut T) -> Result<(), *mut T> {
        // Check that the ptr is not null.
        if ptr.is_null() {
            return Err(ptr);
        }

        let r = self.ptr.compare_exchange(
            core::ptr::null_mut(),
            ptr,
            Ordering::SeqCst,
            Ordering::Relaxed,
        );

        match r {
            Ok(_) => Ok(()),
            Err(x) => Err(x),
        }
    }

    /// The return value is a non-null pointer.
    /// returns error if the internal pointer is NULL.
    fn load(&self) -> Result<*mut T, ()> {
        let p = self.ptr.load(Ordering::Relaxed);
        if p.is_null() { Err(()) } else { Ok(p) }
    }
}
