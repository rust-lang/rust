//! Ensure we report UB when the buffer is smaller than 16 bytes (even if the thread
//! name would fit in the smaller buffer).
//@only-target: android  # Miri supports prctl for Android only

fn main() {
    let mut buf = vec![0u8; 15];
    unsafe {
        libc::prctl(libc::PR_GET_NAME, buf.as_mut_ptr().cast::<libc::c_char>()); //~ ERROR: memory access failed: expected a pointer to 16 bytes of memory, but got alloc952 which is only 15 bytes from the end of the allocation
    }
}
