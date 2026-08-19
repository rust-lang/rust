use crate::io::BorrowedCursor;

pub fn fill_buf(mut cursor: BorrowedCursor<'_, u8>) {
    while cursor.capacity() != 0 {
        let res = unsafe {
            hermit_abi::read_entropy(cursor.as_mut().as_mut_ptr().cast(), cursor.capacity(), 0)
        };
        assert_ne!(res, -1, "failed to generate random data");
        // SAFETY: We've just initialized `res` bytes.
        unsafe {
            cursor.advance(res as usize);
        }
    }
}
