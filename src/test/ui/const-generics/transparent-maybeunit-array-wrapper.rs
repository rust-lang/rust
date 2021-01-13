// run-pass
// revisions: full min

#![cfg_attr(full, feature(const_generics))]
#![cfg_attr(full, allow(incomplete_features))]

use std::mem::MaybeUninit;

#[repr(transparent)]
pub struct MaybeUninitWrapper<const N: usize>(MaybeUninit<[u64; N]>);

fn main() {}
