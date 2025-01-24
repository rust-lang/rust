extern "C" {
    fn TEE_GenerateRandom(randomBuffer: *mut core::ffi::c_void, randomBufferLen: libc::size_t);
}

pub fn fill_bytes(bytes: &mut [u8]) {
    unsafe { TEE_GenerateRandom(bytes.as_mut_ptr().cast(), bytes.len()) }
}
