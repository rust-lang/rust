//! Garbage collection of interned values.
//!
//! The GC is a simple mark-and-sweep GC: you first mark all storages, then the
//! GC visits them, and each live value they refer, recursively, then removes
//! those not marked. The sweep phase is done in parallel.

use std::{hash::Hash, marker::PhantomData, ops::ControlFlow};

use dashmap::DashMap;
use hashbrown::raw::RawTable;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use rustc_hash::{FxBuildHasher, FxHashSet};
use triomphe::{Arc, ThinArc};

use crate::{Internable, InternedRef, InternedSliceRef, SliceInternable};

trait Storage {
    fn len(&self) -> usize;

    fn mark(&self, gc: &mut GarbageCollector);

    fn sweep(&self, gc: &GarbageCollector);
}

struct InternedStorage<T>(PhantomData<fn() -> T>);

impl<T: Internable + GcInternedVisit> Storage for InternedStorage<T> {
    fn len(&self) -> usize {
        T::storage().get().len()
    }

    fn mark(&self, gc: &mut GarbageCollector) {
        let storage = T::storage().get();
        for item in storage {
            let item = item.key();
            let addr = Arc::as_ptr(item).addr();
            if Arc::strong_count(item) > 1 {
                // The item is referenced from the outside.
                gc.alive.insert(addr);
                item.visit_with(gc);
            }
        }
    }

    fn sweep(&self, gc: &GarbageCollector) {
        let storage = T::storage().get();
        gc.sweep_storage(storage, |item| item.as_ptr().addr());
    }
}

struct InternedSliceStorage<T>(PhantomData<fn() -> T>);

impl<T: SliceInternable + GcInternedSliceVisit> Storage for InternedSliceStorage<T> {
    fn len(&self) -> usize {
        T::storage().get().len()
    }

    fn mark(&self, gc: &mut GarbageCollector) {
        let storage = T::storage().get();
        for item in storage {
            let item = item.key();
            let addr = ThinArc::as_ptr(item).addr();
            if ThinArc::strong_count(item) > 1 {
                // The item is referenced from the outside.
                gc.alive.insert(addr);
                T::visit_header(&item.header.header, gc);
                T::visit_slice(&item.slice, gc);
            }
        }
    }

    fn sweep(&self, gc: &GarbageCollector) {
        let storage = T::storage().get();
        gc.sweep_storage(storage, |item| item.as_ptr().addr());
    }
}

pub trait GcInternedVisit {
    fn visit_with(&self, gc: &mut GarbageCollector);
}

pub trait GcInternedSliceVisit: SliceInternable {
    fn visit_header(header: &Self::Header, gc: &mut GarbageCollector);
    fn visit_slice(header: &[Self::SliceType], gc: &mut GarbageCollector);
}

#[derive(Default)]
pub struct GarbageCollector {
    alive: FxHashSet<usize>,
    storages: Vec<&'static (dyn Storage + Send + Sync)>,
}

impl GarbageCollector {
    pub fn add_storage<T: Internable + GcInternedVisit>(&mut self) {
        const { assert!(T::USE_GC) };

        self.storages.push(&InternedStorage::<T>(PhantomData));
    }

    pub fn add_slice_storage<T: SliceInternable + GcInternedSliceVisit>(&mut self) {
        const { assert!(T::USE_GC) };

        self.storages.push(&InternedSliceStorage::<T>(PhantomData));
    }

    /// # Safety
    ///
    ///  - This cannot be called if there are some not-yet-recorded type values.
    ///  - All relevant storages must have been added; that is, within the full graph of values,
    ///    the added storages must form a DAG.
    ///  - [`GcInternedVisit`] and [`GcInternedSliceVisit`] must mark all values reachable from the node.
    pub unsafe fn collect(mut self) {
        let total_nodes = self.storages.iter().map(|storage| storage.len()).sum();
        self.alive.clear();
        self.alive.reserve(total_nodes);

        let storages = std::mem::take(&mut self.storages);

        for &storage in &storages {
            storage.mark(&mut self);
        }

        // Miri doesn't support rayon.
        if cfg!(miri) {
            storages.iter().for_each(|storage| storage.sweep(&self));
        } else {
            storages.par_iter().for_each(|storage| storage.sweep(&self));
        }
    }

    pub fn mark_interned_alive<T: Internable>(
        &mut self,
        interned: InternedRef<'_, T>,
    ) -> ControlFlow<()> {
        if interned.strong_count() > 1 {
            // It will be visited anyway, so short-circuit
            return ControlFlow::Break(());
        }
        let addr = interned.as_raw().addr();
        if !self.alive.insert(addr) { ControlFlow::Break(()) } else { ControlFlow::Continue(()) }
    }

    pub fn mark_interned_slice_alive<T: SliceInternable>(
        &mut self,
        interned: InternedSliceRef<'_, T>,
    ) -> ControlFlow<()> {
        if interned.strong_count() > 1 {
            // It will be visited anyway, so short-circuit
            return ControlFlow::Break(());
        }
        let addr = interned.as_raw().addr();
        if !self.alive.insert(addr) { ControlFlow::Break(()) } else { ControlFlow::Continue(()) }
    }

    fn sweep_storage<T: Hash + Eq + Send + Sync>(
        &self,
        storage: &DashMap<T, (), FxBuildHasher>,
        get_addr: impl Fn(&T) -> usize + Send + Sync,
    ) {
        // Miri doesn't support rayon.
        if cfg!(miri) {
            storage.shards().iter().for_each(|shard| {
                self.retain_only_alive(&mut *shard.write(), |item| get_addr(&item.0))
            });
        } else {
            storage.shards().par_iter().for_each(|shard| {
                self.retain_only_alive(&mut *shard.write(), |item| get_addr(&item.0))
            });
        }
    }

    #[inline]
    fn retain_only_alive<T>(&self, map: &mut RawTable<T>, mut get_addr: impl FnMut(&T) -> usize) {
        // This code was copied from DashMap's retain() - which we can't use because we want to run in parallel.
        unsafe {
            // Here we only use `iter` as a temporary, preventing use-after-free
            for bucket in map.iter() {
                let item = bucket.as_mut();
                let addr = get_addr(item);
                if !self.alive.contains(&addr) {
                    map.erase(bucket);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        GarbageCollector, GcInternedSliceVisit, GcInternedVisit, Interned, InternedSliceRef,
    };

    crate::impl_internable!(String);

    #[test]
    fn simple_interned() {
        let a = Interned::new("abc".to_owned());
        let b = Interned::new("abc".to_owned());
        assert_eq!(a, b);
        assert_eq!(a.as_ref(), b.as_ref());
        assert_eq!(a.as_ref(), a.as_ref());
        assert_eq!(a, a.clone());
        assert_eq!(a, a.clone().clone());
        assert_eq!(b.clone(), a.clone().clone());
        assert_eq!(*a, "abc");
        assert_eq!(*b, "abc");
        assert_eq!(b.as_ref().to_owned(), a);
        let c = Interned::new("def".to_owned());
        assert_ne!(a, c);
        assert_ne!(b, c);
        assert_ne!(b.as_ref(), c.as_ref());
        assert_eq!(*c.as_ref(), "def");
        drop(c);
        assert_eq!(*a, "abc");
        assert_eq!(*b, "abc");
        drop(a);
        assert_eq!(*b, "abc");
        drop(b);
    }

    #[test]
    fn simple_gc() {
        #[derive(Debug, PartialEq, Eq, Hash)]
        struct GcString(String);

        crate::impl_internable!(gc; GcString);

        impl GcInternedVisit for GcString {
            fn visit_with(&self, _gc: &mut GarbageCollector) {}
        }

        crate::impl_slice_internable!(gc; StringSlice, String, u32);
        type InternedSlice = crate::InternedSlice<StringSlice>;

        impl GcInternedSliceVisit for StringSlice {
            fn visit_header(_header: &Self::Header, _gc: &mut GarbageCollector) {}

            fn visit_slice(_header: &[Self::SliceType], _gc: &mut GarbageCollector) {}
        }

        let (a, d) = {
            let a = Interned::new_gc(GcString("abc".to_owned())).to_owned();
            let b = Interned::new_gc(GcString("abc".to_owned())).to_owned();
            assert_eq!(a, b);
            assert_eq!(a.as_ref(), b.as_ref());
            assert_eq!(a.as_ref(), a.as_ref());
            assert_eq!(a, a.clone());
            assert_eq!(a, a.clone().clone());
            assert_eq!(b.clone(), a.clone().clone());
            assert_eq!(a.0, "abc");
            assert_eq!(b.0, "abc");
            assert_eq!(b.as_ref().to_owned(), a);
            let c = Interned::new_gc(GcString("def".to_owned())).to_owned();
            assert_ne!(a, c);
            assert_ne!(b, c);
            assert_ne!(b.as_ref(), c.as_ref());
            assert_eq!(c.as_ref().0, "def");

            let d = InternedSlice::from_header_and_slice("abc".to_owned(), &[123, 456]);
            let e = InternedSlice::from_header_and_slice("abc".to_owned(), &[123, 456]);
            assert_eq!(d, e);
            assert_eq!(d.to_owned(), e.to_owned());
            assert_eq!(d.header.length, 2);
            assert_eq!(d.header.header, "abc");
            assert_eq!(d.slice, [123, 456]);
            (a, d.to_owned())
        };

        let mut gc = GarbageCollector::default();
        gc.add_slice_storage::<StringSlice>();
        gc.add_storage::<GcString>();
        unsafe { gc.collect() };

        assert_eq!(a.0, "abc");
        assert_eq!(d.header.length, 2);
        assert_eq!(d.header.header, "abc");
        assert_eq!(d.slice, [123, 456]);

        drop(a);
        drop(d);

        let mut gc = GarbageCollector::default();
        gc.add_slice_storage::<StringSlice>();
        gc.add_storage::<GcString>();
        unsafe { gc.collect() };
    }

    #[test]
    fn gc_visit() {
        #[derive(PartialEq, Eq, Hash)]
        struct GcInterned(InternedSliceRef<'static, StringSlice>);

        crate::impl_internable!(gc; GcInterned);

        impl GcInternedVisit for GcInterned {
            fn visit_with(&self, gc: &mut GarbageCollector) {
                _ = gc.mark_interned_slice_alive(self.0);
            }
        }

        crate::impl_slice_internable!(gc; StringSlice, String, i32);
        type InternedSlice = crate::InternedSlice<StringSlice>;

        impl GcInternedSliceVisit for StringSlice {
            fn visit_header(_header: &Self::Header, _gc: &mut GarbageCollector) {}

            fn visit_slice(_header: &[Self::SliceType], _gc: &mut GarbageCollector) {}
        }

        let outer = {
            let inner = InternedSlice::from_header_and_slice("abc".to_owned(), &[123, 456, 789]);
            Interned::new_gc(GcInterned(inner)).to_owned()
        };

        let mut gc = GarbageCollector::default();
        gc.add_slice_storage::<StringSlice>();
        gc.add_storage::<GcInterned>();
        unsafe { gc.collect() };

        assert_eq!(outer.0.header.header, "abc");
        assert_eq!(outer.0.slice, [123, 456, 789]);

        drop(outer);

        let mut gc = GarbageCollector::default();
        gc.add_slice_storage::<StringSlice>();
        gc.add_storage::<GcInterned>();
        unsafe { gc.collect() };
    }
}
