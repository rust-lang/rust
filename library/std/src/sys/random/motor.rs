use crate::io::BorrowedCursor;

// https://github.com/moturus/motor-os/issues/49
pub fn fill_buf(mut cursor: BorrowedCursor<'_, u8>) {
    moto_rt::fill_random_bytes(cursor.ensure_init());
    // SAFETY: We've just initialized all the bytes
    unsafe {
        cursor.advance(cursor.capacity());
    }
}
