use crate::mem;

pub type Key = u32;

// Callback invoked when a thread local variable is destroyed (either when the thread
// exits or when the thread local variable is destroyed)
#[no_mangle]
#[inline(never)]
pub extern "C" fn _thread_local_destroy(user_data_low: i32, user_data_high: i32, val_low: i32, val_high: i32) {
    let user_data_low: u64 = (user_data_low as u64) & 0xFFFFFFFF;
    let user_data_high: u64 = (user_data_high as u64) << 32;
    let user_data = user_data_low + user_data_high;

    let val_low: u64 = (val_low as u64) & 0xFFFFFFFF;
    let val_high: u64 = (val_high as u64) << 32;
    let val = val_low + val_high;

    if user_data != 0 {
        unsafe {
            #[cfg(target_arch = "wasm32")]
            let user_data = user_data as u32;
            let dtor: unsafe extern "C" fn(*mut u8) = mem::transmute(user_data);
            let value = val as *mut u8;
            dtor(value);
        }
    }
}

#[inline]
pub unsafe fn create(dtor: Option<unsafe extern "C" fn(*mut u8)>) -> Key {
    let user_data = match dtor {
        Some(a) => a as u64,
        None => 0,
    };
    wasi::thread_local_create(user_data).unwrap()
}

#[inline]
pub unsafe fn set(key: Key, value: *mut u8) {
    let value = value as u64;
    wasi::thread_local_set(key, value).unwrap();
}

#[inline]
pub unsafe fn get(key: Key) -> *mut u8 {
    let value = wasi::thread_local_get(key).unwrap();
    value as *mut u8
}

#[inline]
pub unsafe fn destroy(key: Key) {
    let _ = wasi::thread_local_destroy(key);
}

#[inline]
pub fn requires_synchronized_create() -> bool {
    false
}
