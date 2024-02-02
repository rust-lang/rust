use crate::alloc::{alloc, Layout};

pub type Key = usize;

#[inline]
pub unsafe fn create(_dtor: Option<unsafe extern "C" fn(*mut u8)>) -> Key {
    alloc(Layout::new::<*mut u8>()) as _
}

#[inline]
pub unsafe fn set(key: Key, value: *mut u8) {
    let key: *mut *mut u8 = core::ptr::from_exposed_addr_mut(key);
    *key = value;
}

#[inline]
pub unsafe fn get(key: Key) -> *mut u8 {
    let key: *mut *mut u8 = core::ptr::from_exposed_addr_mut(key);
    *key
}

#[inline]
pub unsafe fn destroy(_key: Key) {}
