use core::intrinsics;
use core::mem;
use core::ptr;

/// This replaces the value behind the `v` unique reference by calling the
/// relevant function.
///
/// If a panic occurs in the `change` closure, the entire process will be aborted.
#[allow(dead_code)] // keep as illustration and for future use
#[inline]
pub fn take_mut<T>(v: &mut T, change: impl FnOnce(T) -> T) {
    replace(v, |value| (change(value), ()))
}

/// This replaces the value behind the `v` unique reference by calling the
/// relevant function, and returns a result obtained along the way.
///
/// If a panic occurs in the `change` closure, the entire process will be aborted.
#[inline]
pub fn replace<T, R>(v: &mut T, change: impl FnOnce(T) -> (T, R)) -> R {
    struct PanicGuard;
    impl Drop for PanicGuard {
        fn drop(&mut self) {
            intrinsics::abort()
        }
    }
    let guard = PanicGuard;
    let value = unsafe { ptr::read(v) };
    let (new_value, ret) = change(value);
    unsafe {
        ptr::write(v, new_value);
    }
    mem::forget(guard);
    ret
}
