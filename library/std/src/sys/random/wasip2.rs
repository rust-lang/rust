pub fn fill_bytes(bytes: &mut [u8]) {
    bytes.copy_from_slice(&wasip2::random::random::get_random_bytes(
        u64::try_from(bytes.len()).unwrap(),
    ));
}

pub fn hashmap_random_keys() -> (u64, u64) {
    wasip2::random::insecure_seed::insecure_seed()
}
