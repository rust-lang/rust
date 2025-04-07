pub fn fill_bytes(mut bytes: &mut [u8]) {
    while !bytes.is_empty() {
        let r = unsafe { libc::getrandom(bytes.as_mut_ptr().cast(), bytes.len(), 0) };
        assert_ne!(r, -1, "failed to generate random data");
        bytes = &mut bytes[r as usize..];
    }
}
