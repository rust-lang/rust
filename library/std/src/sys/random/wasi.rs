// Rely on wasi-libc to fill a buffer with random bytes as that's got all the
// delegation necessary for per-platform bindings.
unsafe extern "C" {
    fn __wasilibc_random(buf: *mut u8, buf_len: usize) -> i32;
}

pub fn fill_bytes(bytes: &mut [u8]) {
    unsafe {
        __wasilibc_random(bytes.as_mut_ptr(), bytes.len());
    }
}

// Use the `insecure_see` import specifically for wasip2/wasip3
#[cfg(target_os = "wasi")]
pub fn hashmap_random_keys() -> (u64, u64) {
    cfg_select! {
        target_env = "p2" => wasip2::random::insecure_seed::insecure_seed(),
        target_env = "p3" => wasip3::random::insecure_seed::get_insecure_seed(),
    }
}
