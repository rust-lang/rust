#[cfg(target_env = "p2")]
use wasip2::random::{insecure_seed::insecure_seed as get_insecure_seed, random::get_random_bytes};
#[cfg(target_env = "p3")]
use wasip3::random::{insecure_seed::get_insecure_seed, random::get_random_bytes};

use crate::io::BorrowedCursor;

pub fn fill_buf(mut cursor: BorrowedCursor<'_, u8>) {
    cursor.append(&get_random_bytes(u64::try_from(cursor.capacity()).unwrap()));
}

pub fn hashmap_random_keys() -> (u64, u64) {
    get_insecure_seed()
}
