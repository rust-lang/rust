//! Garbage collection of interned values.

use std::{marker::PhantomData, ops::ControlFlow};

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
        if cfg!(miri) {
            storage.shards().iter().for_each(|shard| {
                gc.retain_only_alive(&mut *shard.write(), |item| item.0.as_ptr().addr())
            });
        } else {
            storage.shards().par_iter().for_each(|shard| {
                gc.retain_only_alive(&mut *shard.write(), |item| item.0.as_ptr().addr())
            });
        }
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
        if cfg!(miri) {
            storage.shards().iter().for_each(|shard| {
                gc.retain_only_alive(&mut *shard.write(), |item| item.0.as_ptr().addr())
            });
        } else {
            storage.shards().par_iter().for_each(|shard| {
                gc.retain_only_alive(&mut *shard.write(), |item| item.0.as_ptr().addr())
            });
        }
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
    storages: Vec<Box<dyn Storage + Send + Sync>>,
}

impl GarbageCollector {
    pub fn add_storage<T: Internable + GcInternedVisit>(&mut self) {
        const { assert!(T::USE_GC) };

        self.storages.push(Box::new(InternedStorage::<T>(PhantomData)));
    }

    pub fn add_slice_storage<T: SliceInternable + GcInternedSliceVisit>(&mut self) {
        const { assert!(T::USE_GC) };

        self.storages.push(Box::new(InternedSliceStorage::<T>(PhantomData)));
    }

    /// # Safety
    ///
    /// This cannot be called if there are some not-yet-recorded type values.
    pub unsafe fn collect(mut self) {
        let total_nodes = self.storages.iter().map(|storage| storage.len()).sum();
        self.alive = FxHashSet::with_capacity_and_hasher(total_nodes, FxBuildHasher);

        let storages = std::mem::take(&mut self.storages);

        for storage in &storages {
            storage.mark(&mut self);
        }

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

    #[inline]
    fn retain_only_alive<T>(&self, map: &mut RawTable<T>, mut get_addr: impl FnMut(&T) -> usize) {
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

        crate::impl_slice_internable!(gc; StringSlice, String, String);
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

            let d = InternedSlice::from_header_and_slice(
                "abc".to_owned(),
                &["def".to_owned(), "123".to_owned()],
            );
            let e = InternedSlice::from_header_and_slice(
                "abc".to_owned(),
                &["def".to_owned(), "123".to_owned()],
            );
            assert_eq!(d, e);
            assert_eq!(d.to_owned(), e.to_owned());
            assert_eq!(d.header.length, 2);
            assert_eq!(d.header.header, "abc");
            assert_eq!(d.slice, ["def", "123"]);
            (a, d.to_owned())
        };

        let mut gc = GarbageCollector::default();
        gc.add_slice_storage::<StringSlice>();
        gc.add_storage::<GcString>();
        unsafe { gc.collect() };

        assert_eq!(a.0, "abc");
        assert_eq!(d.header.length, 2);
        assert_eq!(d.header.header, "abc");
        assert_eq!(d.slice, ["def", "123"]);

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
