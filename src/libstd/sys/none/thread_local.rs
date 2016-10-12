use alloc::boxed::Box;

struct TlsKey {
	value: *mut u8,
	dtor: Option<unsafe extern fn(*mut u8)>,
}

pub type Key = usize;

impl From<TlsKey> for Key {
	fn from(k: TlsKey) -> Key {
		Box::into_raw(Box::new(k)) as usize
	}
}

impl From<Key> for TlsKey {
	fn from(k: Key) -> TlsKey {
		unsafe { *Box::from_raw(k as *mut TlsKey) }
	}
}

fn borrow_key(k: &Key) -> &mut TlsKey {
	unsafe { &mut*(*k as *mut TlsKey) }
}

#[inline]
pub unsafe fn create(dtor: Option<unsafe extern fn(*mut u8)>) -> Key {
	TlsKey { value: ::ptr::null_mut(), dtor: dtor }.into()
}

#[inline]
pub unsafe fn set(key: Key, value: *mut u8) {
	borrow_key(&key).value=value
}

#[inline]
pub unsafe fn get(key: Key) -> *mut u8 {
    borrow_key(&key).value
}

#[inline]
pub unsafe fn destroy(key: Key) {
	match TlsKey::from(key) {
		TlsKey{value, dtor: Some(dtor)} if value!=::ptr::null_mut() => dtor(value),
		_ => {},
	}
}
