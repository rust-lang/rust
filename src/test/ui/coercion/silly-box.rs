// build-pass
#![feature(coerce_unsized, ptr_metadata, unsize)]

use std::marker::Unsize;
use std::ops::CoerceUnsized;
use std::ptr::{NonNull, TypedMetadata};

struct SillyBox<T: ?Sized> {
    data: NonNull<()>,
    meta: TypedMetadata<T>,
}

impl<T: ?Sized + Unsize<U>, U: ?Sized> CoerceUnsized<SillyBox<U>> for SillyBox<T> {}

fn do_unsize_slice(it: SillyBox<[u8; 5]>) -> SillyBox<[u8]> {
    it
}

struct S;
trait Trait {}
impl Trait for S {}

fn do_unsize_trait(it: SillyBox<S>) -> SillyBox<dyn Trait> {
    it
}

fn main() {}
