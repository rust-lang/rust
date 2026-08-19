use crate::io::BorrowedCursor;
use crate::sys::pal::abi;

pub fn fill_buf(mut cursor: BorrowedCursor<'_, u8>) {
    let result = unsafe {
        abi::SOLID_RNG_SampleRandomBytes(cursor.as_mut().as_mut_ptr().cast(), cursor.capacity())
    };
    assert_eq!(result, 0, "failed to generate random data");
    // SAFETY: We've just initialized all the bytes with random data
    unsafe {
        cursor.advance(cursor.capacity());
    }
}
