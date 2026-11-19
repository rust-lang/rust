//@ compile-flags: -Znext-solver
//@ check-pass

// An unsafe binder has the layout of its inner type, niches included.

#![feature(unsafe_binders, ptr_metadata)]
#![allow(incomplete_features)]

use std::fmt::Debug;
use std::mem::{ManuallyDrop as MD, align_of, size_of};
use std::num::NonZero;
use std::ptr::DynMetadata;

const _: () = assert!(size_of::<unsafe<'a> &'a u8>() == size_of::<&u8>());
const _: () = assert!(size_of::<Option<unsafe<'a> &'a u8>>() == size_of::<&u8>());
const _: () = assert!(size_of::<Option<unsafe<> NonZero<u32>>>() == 4);
const _: () = assert!(align_of::<unsafe<'a> (&'a u8, u16)>() == align_of::<(&u8, u16)>());

// The vtable niche survives a binder in the metadata type and in the pointee.
type BoundMeta = DynMetadata<unsafe<'a> MD<dyn Debug + 'a>>;
const _: () = assert!(size_of::<Option<BoundMeta>>() == size_of::<BoundMeta>());
const _: () = assert!(
    size_of::<Option<*const unsafe<'a> MD<dyn Debug + 'a>>>()
        == size_of::<*const unsafe<'a> MD<dyn Debug + 'a>>()
);

fn main() {}
