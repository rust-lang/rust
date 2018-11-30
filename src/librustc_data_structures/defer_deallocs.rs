// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::ptr::NonNull;
use std::alloc::{Alloc, Global, Layout};
use std::collections::{HashMap, HashSet};
use std::hash::{Hash, BuildHasher, BuildHasherDefault};
use std::mem::{self, needs_drop};

#[derive(Default)]
pub struct DeferredDeallocs {
    allocations: Vec<(NonNull<u8>, Layout)>,
}

impl DeferredDeallocs {
    #[inline]
    pub fn add(&mut self, ptr: NonNull<u8>, layout: Layout) {
        self.allocations.push((ptr, layout));
    }
}

impl Drop for DeferredDeallocs {
    fn drop(&mut self) {
        for &(ptr, layout) in &self.allocations {
            unsafe {
                Global.dealloc(ptr, layout);
            }
        }
    }
}

pub unsafe trait DeferDeallocs {
    fn defer(&self, deferred: &mut DeferredDeallocs);
}

#[macro_export]
macro_rules! impl_defer_dellocs_for_no_drop_type {
    ([$($p:tt)*] $t:ty) => {
        unsafe impl $($p)* $crate::defer_deallocs::DeferDeallocs for $t {
            #[inline(always)]
            fn defer(&self, _: &mut $crate::defer_deallocs::DeferredDeallocs) {
                assert!(!::std::mem::needs_drop::<Self>());
            }
        }
    }
}

unsafe impl<T: DeferDeallocs> DeferDeallocs for Vec<T> {
    #[inline]
    fn defer(&self, deferred: &mut DeferredDeallocs) {
        if mem::size_of::<T>() == 0 || self.capacity() == 0 {
            return;
        }
        let ptr = unsafe {
            NonNull::new_unchecked(self.as_ptr() as *mut u8)
        };
        let size = mem::size_of::<T>();
        let align = mem::align_of::<T>();
        let layout = unsafe {
            Layout::from_size_align_unchecked(self.capacity() * size, align)
        };
        deferred.add(ptr, layout);
        if needs_drop::<T>() {
            for v in self {
                v.defer(deferred)
            }
        }
    }
}

impl_defer_dellocs_for_no_drop_type!([] ());
impl_defer_dellocs_for_no_drop_type!([] bool);
impl_defer_dellocs_for_no_drop_type!([] usize);
impl_defer_dellocs_for_no_drop_type!([] u32);
impl_defer_dellocs_for_no_drop_type!([] u64);
impl_defer_dellocs_for_no_drop_type!([<T>] BuildHasherDefault<T>);
impl_defer_dellocs_for_no_drop_type!([<'a, T: ?Sized>] &'a mut T);
impl_defer_dellocs_for_no_drop_type!([<'a, T: ?Sized>] &'a T);

unsafe impl<T: DeferDeallocs> DeferDeallocs for [T] {
    #[inline]
    fn defer(&self, deferred: &mut DeferredDeallocs) {
        for v in self {
            v.defer(deferred)
        }
    }
}

unsafe impl<T: DeferDeallocs> DeferDeallocs for Option<T> {
    #[inline]
    fn defer(&self, deferred: &mut DeferredDeallocs) {
        self.as_ref().map(|v| v.defer(deferred));
    }
}

unsafe impl<
    T1: DeferDeallocs,
    T2: DeferDeallocs,
> DeferDeallocs
for (T1, T2) {
    #[inline]
    fn defer(&self, deferred: &mut DeferredDeallocs) {
        self.0.defer(deferred);
        self.1.defer(deferred);
    }
}

unsafe impl<
    T1: DeferDeallocs,
    T2: DeferDeallocs,
    T3: DeferDeallocs,
> DeferDeallocs
for (T1, T2, T3) {
    #[inline]
    fn defer(&self, deferred: &mut DeferredDeallocs) {
        self.0.defer(deferred);
        self.1.defer(deferred);
        self.2.defer(deferred);
    }
}

unsafe impl<T: DeferDeallocs + ?Sized> DeferDeallocs for Box<T> {
    #[inline]
    fn defer(&self, deferred: &mut DeferredDeallocs) {
        let size = mem::size_of_val(&**self);
        // FIXME: What happens when `size` is 0 here? Like [T] when T is zero sized
        if size == 0 {
            return;
        }
        let layout = unsafe {
            Layout::from_size_align_unchecked(size, mem::align_of_val(&**self))
        };
        deferred.add(unsafe {
            NonNull::new_unchecked(&**self as *const T as *mut u8)
        }, layout);
        (**self).defer(deferred);
    }
}

unsafe impl<
    K: DeferDeallocs + Eq + Hash,
    V: DeferDeallocs,
    S: DeferDeallocs + BuildHasher
> DeferDeallocs
for HashMap<K, V, S> {
    #[inline]
    fn defer(&self, deferred: &mut DeferredDeallocs) {
        self.hasher().defer(deferred);
        if let Some((ptr, layout)) = self.raw_alloc() {
            deferred.add(ptr, layout);
        }
        if needs_drop::<(K, V)>() {
            for (k, v) in self.iter() {
                k.defer(deferred);
                v.defer(deferred);
            }
        }
    }
}

unsafe impl<T: DeferDeallocs + Eq + Hash, S: DeferDeallocs + BuildHasher> DeferDeallocs
for HashSet<T, S> {
    #[inline]
    fn defer(&self, deferred: &mut DeferredDeallocs) {
        self.hasher().defer(deferred);
        if let Some((ptr, layout)) = self.raw_alloc() {
            deferred.add(ptr, layout);
        }
        if needs_drop::<T>() {
            for v in self.iter() {
                v.defer(deferred);
            }
        }
    }
}
