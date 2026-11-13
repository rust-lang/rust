//! Regression test for <https://github.com/rust-lang/rust/issues/81162>.
//@ check-pass

use std::convert::TryFrom;
use std::fmt::Debug;

fn read<T>() -> Vec<T>
where
    for<'a> T: TryFrom<&'a [f32]>,
    for<'a> <T as TryFrom<&'a [f32]>>::Error: Debug,
{
    Vec::new()
}

fn main() {
    let _ = read::<[f32; 128]>();
}
