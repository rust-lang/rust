use crate::boxed::Box;
use crate::ptr;

pub type Key = usize;

struct Allocated {
    value: *mut u8,
    dtor: Option<unsafe extern fn(*mut u8)>,
}

#[inline]
pub unsafe fn create(dtor: Option<unsafe extern fn(*mut u8)>) -> Key {
    Box::into_raw(Box::new(Allocated {
        value: ptr::null_mut(),
        dtor,
    })) as usize
}

#[inline]
pub unsafe fn set(key: Key, value: *mut u8) {
    (*(key as *mut Allocated)).value = value;
}

#[inline]
pub unsafe fn get(key: Key) -> *mut u8 {
    (*(key as *mut Allocated)).value
}

#[inline]
pub unsafe fn destroy(key: Key) {
    let key = Box::from_raw(key as *mut Allocated);
    if let Some(f) = key.dtor {
        f(key.value);
    }
}

#[inline]
pub fn requires_synchronized_create() -> bool {
    false
}
