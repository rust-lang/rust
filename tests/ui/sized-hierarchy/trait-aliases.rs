//@ check-pass
//@ compile-flags: --crate-type=lib
#![feature(trait_alias)]

// Tests that `?Sized` is migrated to `MetaSized` in trait aliases.

use std::ops::{Index, IndexMut};

pub trait SlicePrereq<T> = ?Sized + IndexMut<usize, Output = <[T] as Index<usize>>::Output>;
