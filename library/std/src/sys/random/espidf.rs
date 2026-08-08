use crate::ffi::c_void;
use crate::io::BorrowedCursor;

unsafe extern "C" {
    fn esp_fill_random(buf: *mut c_void, len: usize);
}

pub fn fill_buf(&mut self, mut cursor: BorrowedCursor<'_, u8>) {
    unsafe {
        esp_fill_random(cursor.as_mut().as_mut_ptr().cast(), cursor.capacity());
        cursor.advance(cursor.capacity());
    }
}
