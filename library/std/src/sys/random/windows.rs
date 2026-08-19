use crate::io::BorrowedCursor;
use crate::sys::c;

#[cfg(not(target_vendor = "win7"))]
#[inline]
pub fn fill_buf(mut cursor: BorrowedCursor<'_, u8>) {
    let ret = unsafe { c::ProcessPrng(cursor.as_mut().as_mut_ptr().cast(), cursor.capacity()) };
    // ProcessPrng is documented as always returning `TRUE`.
    // https://learn.microsoft.com/en-us/windows/win32/seccng/processprng#return-value
    debug_assert_eq!(ret, c::TRUE);
}

#[cfg(target_vendor = "win7")]
pub fn fill_buf(mut cursor: BorrowedCursor<'_, u8>) {
    while cursor.capacity() != 0 {
        let len = cursor.capacity().try_into().unwrap_or(u32::MAX);
        let ret = unsafe { c::RtlGenRandom(cursor.as_mut().as_mut_ptr().cast(), len) };
        assert!(ret, "failed to generate random data");
        // SAFETY: We've just initialized `len` bytes
        unsafe {
            cursor.advance(len as usize);
        }
    }
}
