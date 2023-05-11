// run-pass
// Test that fn item types are zero-sized.

use std::mem::{size_of, size_of_val};

fn main() {
    assert_eq!(size_of_val(&main), 0);

    let (a, b) = (size_of::<u8>, size_of::<u16>);
    assert_eq!(size_of_val(&a), 0);
    assert_eq!(size_of_val(&b), 0);
    assert_eq!((a(), b()), (1, 2));
}
