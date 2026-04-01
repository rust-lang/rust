unsafe extern "C" {
    fn trusty_rng_secure_rand(randomBuffer: *mut core::ffi::c_void, randomBufferLen: libc::size_t);
}

pub fn fill_bytes(bytes: &mut [u8]) {
    unsafe { trusty_rng_secure_rand(bytes.as_mut_ptr().cast(), bytes.len()) }
}
