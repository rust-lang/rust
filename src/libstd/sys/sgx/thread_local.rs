use super::abi::tls::{Tls, Key as AbiKey};

pub type Key = usize;

#[inline]
pub unsafe fn create(dtor: Option<unsafe extern fn(*mut u8)>) -> Key {
    Tls::create(dtor).as_usize()
}

#[inline]
pub unsafe fn set(key: Key, value: *mut u8) {
    Tls::set(AbiKey::from_usize(key), value)
}

#[inline]
pub unsafe fn get(key: Key) -> *mut u8 {
    Tls::get(AbiKey::from_usize(key))
}

#[inline]
pub unsafe fn destroy(key: Key) {
    Tls::destroy(AbiKey::from_usize(key))
}

#[inline]
pub fn requires_synchronized_create() -> bool {
    false
}
