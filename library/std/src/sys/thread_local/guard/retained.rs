//! Guards which run all TLS destructors themselves, and only then
//! [`crate::rt::thread_cleanup`], never free the values of keys without a
//! destructor before the cleanup. Hence, local pointers need no destructor.

pub const LOCAL_POINTER_NEEDS_DTOR: bool = false;

/// Only referenced by the `keep` destructor of `local_pointer!`, which is
/// never registered here (`LOCAL_POINTER_NEEDS_DTOR` is `false`).
pub fn cleanup_pending() -> bool {
    rtabort!("`cleanup_pending` called although `LOCAL_POINTER_NEEDS_DTOR` is false")
}
