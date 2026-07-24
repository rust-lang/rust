//@ run-pass
//@ compile-flags: -C debug-assertions

// This is a regression test for https://github.com/rust-lang/rust/issues/159433

use std::num::NonZeroI32;

type Thing = Result<(), NonZeroI32>;

fn main() {
    let _ = unsafe { std::mem::transmute::<Thing, Thing>(Err(NonZeroI32::new(-1).unwrap())) };
}
