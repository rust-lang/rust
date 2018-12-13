// compile-pass

#![feature(const_nonzero_methods)]

use std::num::NonZeroU8;

const X: NonZeroU8 = unsafe { NonZeroU8::new_unchecked(5) };
const Y: u8 = X.get();

fn main() {
}
