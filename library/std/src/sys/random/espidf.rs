use crate::ffi::c_void;

unsafe extern "C" {
    fn esp_fill_random(buf: *mut c_void, len: usize);
}

pub fn fill_bytes(bytes: &mut [u8]) {
    unsafe { esp_fill_random(bytes.as_mut_ptr().cast(), bytes.len()) }
}
