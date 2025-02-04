use crate::ptr;

pub type Key = usize;
type Dtor = unsafe extern "C" fn(*mut u8);

static mut STORAGE: crate::vec::Vec<(*mut u8, Option<Dtor>)> = Vec::new();

#[inline]
pub fn create(dtor: Option<Dtor>) -> Key {
    unsafe {
        #[allow(static_mut_refs)]
        let key = STORAGE.len();
        #[allow(static_mut_refs)]
        STORAGE.push((ptr::null_mut(), dtor));
        key
    }
}

#[inline]
pub unsafe fn set(key: Key, value: *mut u8) {
    unsafe { STORAGE[key].0 = value };
}

#[inline]
pub unsafe fn get(key: Key) -> *mut u8 {
    unsafe { STORAGE[key].0 }
}

#[inline]
pub fn destroy(_key: Key) {}
