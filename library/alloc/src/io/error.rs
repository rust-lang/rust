#[cfg_attr(no_global_oom_handling, expect(unused_imports))]
use crate::{
    boxed::Box,
    io::{Custom, CustomOwner, ErrorKind},
};

#[cfg(not(no_global_oom_handling))]
#[doc(hidden)]
#[unstable(feature = "core_io_internals", reason = "exposed only for libstd", issue = "none")]
pub fn custom_owner_from_box(
    kind: ErrorKind,
    error: Box<dyn core::error::Error + Send + Sync>,
) -> CustomOwner {
    /// # Safety
    ///
    /// `ptr` must be valid to pass into `Box::from_raw`.
    unsafe fn drop_box_raw<T: ?Sized>(ptr: *mut T) {
        // SAFETY
        // Caller ensures `ptr` is valid to pass into `Box::from_raw`.
        drop(unsafe { Box::from_raw(ptr) })
    }

    // SAFETY: the pointer returned by Box::into_raw is non-null.
    let error = unsafe { core::ptr::NonNull::new_unchecked(Box::into_raw(error)) };

    // SAFETY:
    // * `error` is valid up to a static lifetime, and owns its pointee.
    // * `drop_box_raw` is safe to call for the pointer `error` exactly once.
    // * `drop_box_raw` is safe to call on a pointer to this instance of `Custom`,
    //   and will be stored in a `CustomOwner`.
    let custom = unsafe { Custom::from_raw(kind, error, drop_box_raw, drop_box_raw) };

    // SAFETY: the pointer returned by Box::into_raw is non-null.
    let custom = unsafe { core::ptr::NonNull::new_unchecked(Box::into_raw(Box::new(custom))) };

    // SAFETY: the `outer_drop` provided to `custom` is valid for itself.
    unsafe { CustomOwner::from_raw(custom) }
}
