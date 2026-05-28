//! Allows access to the Rayon's thread local value
//! which is preserved when moving jobs across threads

use std::cell::Cell;
use std::ptr;

thread_local!(pub static TLV: Cell<*const ()> = const { Cell::new(ptr::null()) });

#[derive(Copy, Clone)]
pub(crate) struct Tlv(pub(crate) *const ());

impl Tlv {
    #[inline]
    pub(crate) fn null() -> Self {
        Self(ptr::null())
    }
}

unsafe impl Sync for Tlv {}
unsafe impl Send for Tlv {}

/// Sets the current thread-local value
#[inline]
pub(crate) fn set(value: Tlv) {
    TLV.with(|tlv| tlv.set(value.0));
}

/// Returns the current thread-local value
#[inline]
pub(crate) fn get() -> Tlv {
    TLV.with(|tlv| Tlv(tlv.get()))
}
