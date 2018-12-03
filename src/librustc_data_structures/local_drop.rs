// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::collections::{HashMap, HashSet};
use std::hash::BuildHasherDefault;

/// A type which does not look at references in its destructor
/// but only owned data
pub unsafe trait LocalDrop {
    fn check();
}

#[inline(always)]
pub fn check<T: LocalDrop + ?Sized>(_: &T) {
    T::check();
}

trait NoDrop {
    fn check();
}

impl<T: ?Sized> NoDrop for T {
    #[inline(always)]
    default fn check() {}
}
impl<T: Drop + ?Sized> NoDrop for T {
    fn check() {
        panic!(
            "Type {} derives LocalDrop, but has a destructor",
            unsafe { std::intrinsics::type_name::<T>() }
        );
    }
}

#[inline(always)]
pub fn ensure_no_drop<T: ?Sized>() {
    <T as NoDrop>::check();
}

#[macro_export]
macro_rules! impl_trivial_local_drop {
    ([$($p:tt)*] $t:ty) => {
        unsafe impl $($p)* $crate::local_drop::LocalDrop for $t {
            #[inline(always)]
            fn check() {
                assert!(!::std::mem::needs_drop::<Self>());
            }
        }
    }
}

impl_trivial_local_drop!([] bool);
impl_trivial_local_drop!([] u32);
impl_trivial_local_drop!([] u64);
impl_trivial_local_drop!([] usize);
impl_trivial_local_drop!([<T>] BuildHasherDefault<T>);
impl_trivial_local_drop!([<'a, T: 'a + ?Sized>] &'a T);
impl_trivial_local_drop!([<'a, T: 'a + ?Sized>] &'a mut T);

unsafe impl<T: LocalDrop> LocalDrop for Vec<T> {
    #[inline(always)]
    fn check() {
        T::check();
    }
}

unsafe impl<K: LocalDrop, V: LocalDrop, S: LocalDrop> LocalDrop for HashMap<K, V, S> {
    #[inline(always)]
    fn check() {
        K::check();
        V::check();
        S::check();
    }
}

unsafe impl<T: LocalDrop, S: LocalDrop> LocalDrop for HashSet<T, S> {
    #[inline(always)]
    fn check() {
        T::check();
        S::check();
    }
}
