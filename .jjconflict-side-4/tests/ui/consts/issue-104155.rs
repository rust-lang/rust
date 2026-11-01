//@ check-pass

#![allow(forgetting_copy_types)]

const _: () = core::mem::forget(Box::<u32>::default);
const _: () = core::mem::forget(|| Box::<u32>::default());

fn main() {}
