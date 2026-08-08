use crate::io::BorrowedCursor;
use crate::mem::MaybeUninit;
use crate::sys::pal::abi;

pub fn fill_buf(mut cursor: BorrowedCursor<'_, u8>) {
    let bytes = cursor.as_mut();
    let (pre, words, post) = unsafe { bytes.align_to_mut::<MaybeUninit<u32>>() };
    if !words.is_empty() {
        unsafe {
            abi::sys_rand(words.as_mut_ptr().cast(), words.len());
        }
    }

    let mut buf = [0u32; 2];
    let len = (pre.len() + post.len() + size_of::<u32>() - 1) / size_of::<u32>();
    if len != 0 {
        unsafe { abi::sys_rand(buf.as_mut_ptr(), len) };
    }

    let buf = buf.map(u32::to_ne_bytes);
    let buf = buf.as_flattened();
    pre.write_copy_of_slice(&buf[..pre.len()]);
    post.write_copy_of_slice(&buf[pre.len()..pre.len() + post.len()]);

    // SAFETY: We've just initialized all the bytes with random data
    unsafe {
        cursor.advance(cursor.capacity());
    }
}
