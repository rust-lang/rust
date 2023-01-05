// run-pass

// https://github.com/rust-lang/rust/issues/41898

use std::num::NonZeroU64;

fn main() {
    const FOO: NonZeroU64 = unsafe { NonZeroU64::new_unchecked(2) };
    if let FOO = FOO {}
}
