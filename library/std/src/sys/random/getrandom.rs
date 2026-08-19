use crate::io::BorrowedCursor;

pub fn fill_buf(mut cursor: BorrowedCursor<'_, u8>) {
    while cursor.capacity() != 0 {
        let r =
            unsafe { libc::getrandom(cursor.as_mut().as_mut_ptr().cast(), cursor.capacity(), 0) };
        assert_ne!(r, -1, "failed to generate random data");
        // SAFETY: We've just initialized `r` bytes.
        unsafe {
            cursor.advance(r as usize);
        }
    }
}
