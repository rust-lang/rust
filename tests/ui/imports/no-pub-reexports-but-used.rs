//@ check-pass
// https://github.com/rust-lang/rust/issues/115966

mod m {
    pub(crate) type A = u8;
}

#[warn(unused_imports)] //~ NOTE: the lint level is defined here
pub use m::*;
//~^ WARNING: glob import doesn't reexport anything with visibility `pub` because no imported item is public enough
//~| NOTE: the most public imported item is `pub(crate)`

fn main() {
    let _: A;
}
