// Miri currently doesn't require types without drop glue to be
// valid when dropped. This test confirms that behavior.
// This is not a stable guarantee!

use std::ptr;

fn main() {
    let mut not_a_bool = 13u8;
    unsafe { ptr::drop_in_place(&mut not_a_bool as *mut u8 as *mut bool) };
}
