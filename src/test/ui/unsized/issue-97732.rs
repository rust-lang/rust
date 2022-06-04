// check-pass

#![feature(coerce_unsized)]

// Ensure that unsizing structs that contain ZSTs at non-zero offsets don't ICE

use std::ops::CoerceUnsized;

#[repr(C)]
pub struct BoxWithZstTail<T: ?Sized>(Box<T>, ());

impl<S: ?Sized, T: ?Sized> CoerceUnsized<BoxWithZstTail<T>> for BoxWithZstTail<S> where
    Box<S>: CoerceUnsized<Box<T>>
{
}

pub fn noop_dyn_upcast_with_zst_tail(
    b: BoxWithZstTail<dyn Send + Sync>,
) -> BoxWithZstTail<dyn Send> {
    b
}

fn main() {}
