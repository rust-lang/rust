// run-pass

use std::num::NonZeroU8;

const X: NonZeroU8 = unsafe { NonZeroU8::new_unchecked(5) };
const Y: u8 = X.get();

const ZERO: Option<NonZeroU8> = NonZeroU8::new(0);
const ONE: Option<NonZeroU8> = NonZeroU8::new(1);

fn main() {
    assert_eq!(Y, 5);

    assert!(ZERO.is_none());
    assert_eq!(ONE.unwrap().get(), 1);
}
