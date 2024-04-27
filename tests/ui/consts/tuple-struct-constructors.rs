//@ run-pass
//
// https://github.com/rust-lang/rust/issues/41898

use std::num::NonZero;

fn main() {
    const FOO: NonZero<u64> = unsafe { NonZero::new_unchecked(2) };
    if let FOO = FOO {}
}
