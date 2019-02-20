// The following functions are needed by libunwind. These symbols are named
// in pre-link args for the target specification, so keep that in sync.
#[cfg(not(test))]
const EINVAL: i32 = 22;

#[cfg(not(test))]
#[no_mangle]
pub unsafe extern "C" fn __rust_rwlock_rdlock(p: *mut RWLock) -> i32 {
    if p.is_null() {
        return EINVAL;
    }
    (*p).read();
    return 0;
}

#[cfg(not(test))]
#[no_mangle]
pub unsafe extern "C" fn __rust_rwlock_wrlock(p: *mut RWLock) -> i32 {
    if p.is_null() {
        return EINVAL;
    }
    (*p).write();
    return 0;
}
#[cfg(not(test))]
#[no_mangle]
pub unsafe extern "C" fn __rust_rwlock_unlock(p: *mut RWLock) -> i32 {
    if p.is_null() {
        return EINVAL;
    }
    (*p).unlock();
    return 0;
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::array::FixedSizeArray;
    use crate::mem::{self, MaybeUninit};

    // Verify that the bytes of initialized RWLock are the same as in
    // libunwind. If they change, `src/UnwindRustSgx.h` in libunwind needs to
    // be changed too.
    #[test]
    fn test_c_rwlock_initializer() {
        const RWLOCK_INIT: &[u8] = &[
            0x1, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
            0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
            0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
            0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
            0x3, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
            0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
            0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
            0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
            0x1, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
            0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
            0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
            0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
            0x3, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
            0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
            0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
            0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
        ];

        #[inline(never)]
        fn zero_stack() {
            test::black_box(MaybeUninit::<[RWLock; 16]>::zeroed());
        }

        #[inline(never)]
        unsafe fn rwlock_new(init: &mut MaybeUninit<RWLock>) {
            init.write(RWLock::new());
        }

        unsafe {
            // try hard to make sure that the padding/unused bytes in RWLock
            // get initialized as 0. If the assertion below fails, that might
            // just be an issue with the test code and not with the value of
            // RWLOCK_INIT.
            zero_stack();
            let mut init = MaybeUninit::<RWLock>::zeroed();
            rwlock_new(&mut init);
            assert_eq!(
                mem::transmute::<_, [u8; 128]>(init.assume_init()).as_slice(),
                RWLOCK_INIT
            )
        };
    }
}
