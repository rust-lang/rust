use crate::io::BorrowedCursor;

pub fn fill_buf(mut cursor: BorrowedCursor<'_, u8>) {
    unsafe {
        wasip1::random_get(cursor.as_mut().as_mut_ptr().cast(), cursor.capacity())
            .expect("failed to generate random data")
    }
    // SAFETY: We've just initialized all the bytes with random data
    unsafe {
        cursor.advance(cursor.capacity());
    }
}
