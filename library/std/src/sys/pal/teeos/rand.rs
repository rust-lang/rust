pub fn hashmap_random_keys() -> (u64, u64) {
    const KEY_LEN: usize = core::mem::size_of::<u64>();

    let mut v = [0u8; KEY_LEN * 2];
    imp::fill_bytes(&mut v);

    let key1 = v[0..KEY_LEN].try_into().unwrap();
    let key2 = v[KEY_LEN..].try_into().unwrap();

    (u64::from_ne_bytes(key1), u64::from_ne_bytes(key2))
}

mod imp {
    extern "C" {
        fn TEE_GenerateRandom(randomBuffer: *mut core::ffi::c_void, randomBufferLen: libc::size_t);
    }

    pub fn fill_bytes(v: &mut [u8]) {
        unsafe { TEE_GenerateRandom(v.as_mut_ptr() as _, v.len() * crate::mem::size_of::<u8>()) }
    }
}
