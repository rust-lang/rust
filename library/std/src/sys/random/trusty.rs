use crate::io::BorrowedCursor;

unsafe extern "C" {
    fn trusty_rng_secure_rand(randomBuffer: *mut core::ffi::c_void, randomBufferLen: libc::size_t);
}

pub fn fill_buf(mut cursor: BorrowedCursor<'_, u8>) {
    unsafe {
        trusty_rng_secure_rand(cursor.as_mut().as_mut_ptr().cast(), cursor.capacity());
    }
    // SAFETY: We've just initialized all the bytes with random data
    unsafe {
        cursor.advance(cursor.capacity());
    }
}
