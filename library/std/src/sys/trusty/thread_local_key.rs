use crate::ptr;

pub type Key = usize;
type Dtor = unsafe extern "C" fn(*mut u8);

static mut STORAGE: crate::vec::Vec<(*mut u8, Option<Dtor>)> = Vec::new();

#[inline]
pub unsafe fn create(dtor: Option<Dtor>) -> Key {
    let key = STORAGE.len();
    STORAGE.push((ptr::null_mut(), dtor));
    key
}

#[inline]
pub unsafe fn set(key: Key, value: *mut u8) {
    STORAGE[key].0 = value;
}

#[inline]
pub unsafe fn get(key: Key) -> *mut u8 {
    STORAGE[key].0
}

#[inline]
pub unsafe fn destroy(_key: Key) {}

#[inline]
pub fn requires_synchronized_create() -> bool {
    false
}
