// The following functions are needed by libunwind. These symbols are named
// in pre-link args for the target specification, so keep that in sync.

#[cfg(not(test))]
use crate::{
    alloc::{self, Layout},
    lock_api::RawRwLock as _,
    slice, str,
    sync::atomic::Ordering,
};
use crate::{parking_lot::RawRwLock, sync::atomic::AtomicBool};

#[cfg(not(test))]
const EINVAL: i32 = 22;

#[repr(C)]
pub struct RwLock {
    lock: RawRwLock,
    is_write_locked: AtomicBool,
}

#[cfg(not(test))]
#[no_mangle]
pub unsafe extern "C" fn __rust_rwlock_rdlock(p: *mut RwLock) -> i32 {
    if p.is_null() {
        return EINVAL;
    }
    (*p).lock.lock_shared();
    return 0;
}

#[cfg(not(test))]
#[no_mangle]
pub unsafe extern "C" fn __rust_rwlock_wrlock(p: *mut RwLock) -> i32 {
    if p.is_null() {
        return EINVAL;
    }
    (*p).lock.lock_exclusive();
    (*p).is_write_locked.store(true, Ordering::Relaxed);
    return 0;
}
#[cfg(not(test))]
#[no_mangle]
pub unsafe extern "C" fn __rust_rwlock_unlock(p: *mut RwLock) -> i32 {
    if p.is_null() {
        return EINVAL;
    }
    if (*p)
        .is_write_locked
        .compare_exchange(true, false, Ordering::Relaxed, Ordering::Relaxed)
        .is_ok()
    {
        (*p).lock.unlock_exclusive()
    } else {
        (*p).lock.unlock_shared();
    }
    return 0;
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::array::FixedSizeArray;
    use crate::mem::{self, MaybeUninit};

    // Verify that the bytes of an initialized RwLock are the same as in
    // libunwind. If they change, `src/UnwindRustSgx.h` in libunwind needs to
    // be changed too.
    #[test]
    fn test_c_rwlock_initializer() {
        /// The value of a newly initialized `RwLock`. Which happens to be
        /// `RawRwLock::INIT` (a zeroed `usize`), a false boolean (zero)
        /// and then padding.
        const RWLOCK_INIT: &[u8] = &[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];

        #[inline(never)]
        fn zero_stack() {
            test::black_box(MaybeUninit::<[RwLock; 16]>::zeroed());
        }

        #[inline(never)]
        unsafe fn rwlock_new(init: &mut MaybeUninit<RwLock>) {
            use crate::lock_api::RawRwLock as _;
            init.write(RwLock {
                lock: RawRwLock::INIT,
                is_write_locked: AtomicBool::new(false),
            });
        }

        unsafe {
            // try hard to make sure that the padding/unused bytes in RwLock
            // get initialized as 0. If the assertion below fails, that might
            // just be an issue with the test code and not with the value of
            // RWLOCK_INIT.
            zero_stack();
            let mut init = MaybeUninit::<RwLock>::zeroed();
            rwlock_new(&mut init);
            assert_eq!(
                mem::transmute::<_, [u8; 16]>(init.assume_init()).as_slice(),
                RWLOCK_INIT
            )
        };
    }

    #[test]
    fn test_rwlock_memory_layout() {
        assert_eq!(mem::size_of::<RwLock>(), mem::size_of::<usize>() * 2);
        assert_eq!(mem::align_of::<RwLock>(), mem::align_of::<usize>());
    }

    #[test]
    fn test_sgx_on_64bit() {
        #[cfg(target_pointer_width = "32")]
        panic!("The RwLock implementation for SGX only works on 64 bit architectures for now");
    }
}
