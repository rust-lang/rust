use crate::ptr;

pub fn fill_bytes(_: &mut [u8]) {
    panic!("this target does not support random data generation");
}

pub fn hashmap_random_keys() -> (u64, u64) {
    // Use allocation addresses for a bit of randomness. This isn't
    // particularly secure, but there isn't really an alternative.
    let stack = 0u8;
    let heap = Box::new(0u8);
    let k1 = ptr::from_ref(&stack).addr() as u64;
    let k2 = ptr::from_ref(&*heap).addr() as u64;
    (k1, k2)
}
