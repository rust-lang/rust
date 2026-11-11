//@ run-pass
// Tests against a regression surfaced by crater in https://github.com/rust-lang/rust/issues/125193
// Unwind Safety is not a very coherent concept, but we'd prefer no regressions until we kibosh it
// and this is an unstable feature anyways sooo...

use std::panic::UnwindSafe;
use std::task::Context;

fn unwind_safe<T: UnwindSafe>() {}

fn main() {
    unwind_safe::<Context<'_>>(); // test UnwindSafe
    unwind_safe::<&Context<'_>>(); // test RefUnwindSafe
}
