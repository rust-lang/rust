//@ check-pass
//@ compile-flags: --crate-type=lib
#![feature(trait_alias)]

// Checks that `?Sized` in a trait alias doesn't trigger an ICE.

use std::ops::{Index, IndexMut};

pub trait SlicePrereq<T> = ?Sized + IndexMut<usize, Output = <[T] as Index<usize>>::Output>;
