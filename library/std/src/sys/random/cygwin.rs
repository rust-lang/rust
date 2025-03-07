pub fn fill_bytes(mut bytes: &mut [u8]) {
    while !bytes.is_empty() {
        let ret =
            unsafe { libc::getrandom(bytes.as_mut_ptr().cast(), bytes.len(), libc::GRND_NONBLOCK) };
        assert!(ret != -1, "failed to generate random data");
        bytes = &mut bytes[ret as usize..];
    }
}
