// run-pass

use std::num::NonZeroU64;

fn main() {
    const FOO: NonZeroU64 = unsafe { NonZeroU64::new_unchecked(2) };
    if let FOO = FOO {}
}
