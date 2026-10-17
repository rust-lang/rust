//! Guards which run all TLS destructors themselves, and only then
//! [`crate::rt::thread_cleanup`], never free the values of keys without a
//! destructor before the cleanup. Hence, `#[keep_last]` local pointers need no
//! destructor.

pub const KEEP_LAST_NEEDS_DTOR: bool = false;

/// Only referenced by the `keep` destructor of `local_pointer!`, which is
/// never registered here (`KEEP_LAST_NEEDS_DTOR` is `false`).
pub fn cleanup_pending() -> bool {
    rtabort!("`cleanup_pending` called although `KEEP_LAST_NEEDS_DTOR` is false")
}
