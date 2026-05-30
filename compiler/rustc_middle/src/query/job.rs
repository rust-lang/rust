use std::sync::Arc;
use std::{mem, ptr};

use parking_lot::{Condvar, Mutex, MutexGuard};
use rustc_data_structures::sync::{CacheAligned, DynSync, Parker, Unparker};
use rustc_data_structures::{cache_entry, jobserver};
use rustc_span::Span;
use rustc_thread_pool::{Registry, current_thread_index};

use crate::queries::TaggedQueryKey;
use crate::query::Cycle;

pub type QueryJobRef<'a, 'tcx> = &'a QueryJob<'a, 'tcx>;

/// Represents an active query job.
#[derive(Clone, Copy)]
pub struct QueryJob<'a, 'tcx> {
    /// The span corresponding to the reason for which this query was required.
    pub span: Span,

    /// The parent query job which created this job and is implicitly waiting on it.
    pub parent: Option<QueryJobRef<'a, 'tcx>>,

    pub form_tagged_key: &'a (dyn Fn() -> TaggedQueryKey<'tcx> + DynSync),

    pub entry_status: &'a cache_entry::Status,
}

#[derive(Clone, Copy)]
pub struct QueryWaiter<'a, 'tcx> {
    pub span: Span,
    pub parent: Option<QueryJobRef<'a, 'tcx>>,
}

pub struct WorkerParkingArea<'tcx> {
    registry: Arc<rustc_thread_pool::Registry>,
    proxy: Arc<jobserver::Proxy>,
    lots: Box<[CacheAligned<WorkerParkingLot<'tcx>>]>,
}

impl<'tcx> WorkerParkingArea<'tcx> {
    pub fn new(proxy: Arc<jobserver::Proxy>) -> Self {
        let registry = Registry::current();
        let lots =
            (0..registry.num_threads()).map(|_| CacheAligned(WorkerParkingLot::new())).collect();
        Self { registry, proxy, lots }
    }

    #[inline(always)]
    pub fn parker<'a>(&'a self, waiter: QueryWaiter<'a, 'tcx>) -> WorkerParker<'a, 'tcx> {
        WorkerParker { area: self, waiter }
    }
}

#[derive(Clone, Copy)]
pub struct WorkerParker<'a, 'tcx> {
    area: &'a WorkerParkingArea<'tcx>,
    waiter: QueryWaiter<'a, 'tcx>,
}

impl<'a, 'tcx> Parker for WorkerParker<'a, 'tcx> {
    type Interrupt = Cycle<'tcx>;

    fn park(self, validate: impl FnOnce(usize) -> bool) -> Result<(), Self::Interrupt> {
        debug_assert!(Registry::with_current(|registry| ptr::eq(
            &**registry,
            &*self.area.registry
        )));
        let thread_index = current_thread_index().unwrap();
        debug_assert!(thread_index < self.area.registry.num_threads());
        let validate = || validate(thread_index).then(|| self.waiter);
        self.area.lots[thread_index].0.park(validate, &self.area.proxy)
    }
}

impl<'a, 'tcx> Unparker for &'a WorkerParkingArea<'tcx> {
    fn unpark(self, thread_bitmask: u32) {
        debug_assert_eq!(
            thread_bitmask
                & !(u32::MAX >> (u32::BITS - u32::try_from(self.registry.num_threads()).unwrap())),
            0
        );
        let mut waiters = [(); rustc_thread_pool::max_num_threads()].map(|()| None);
        for i in 0..self.registry.num_threads() {
            if thread_bitmask & (1 << i) != 0 {
                waiters[i] = Some(
                    self.lots[i].0.lock_waiter().expect("trying to unpark a non-parked thread"),
                );
            }
        }
        rustc_thread_pool::mark_unblocked(&self.registry, thread_bitmask.count_ones() as usize);
        for waiter in waiters.iter_mut() {
            if let Some(waiter) = waiter.take() {
                waiter.unpark();
            }
        }
    }
}

impl<'tcx> WorkerParkingArea<'tcx> {
    pub fn lock_waiter(&self, thread_index: usize) -> Option<QueryWaiterGuard<'_, 'tcx>> {
        self.lots[thread_index].0.lock_waiter()
    }
}

enum WorkerStatus<'tcx> {
    Free,
    Cycle(Cycle<'tcx>),
    Waiting(QueryWaiter<'tcx, 'tcx>),
}

struct WorkerParkingLot<'tcx> {
    condvar: Condvar,
    status: Mutex<WorkerStatus<'tcx>>,
}

pub struct QueryWaiterGuard<'a, 'tcx> {
    guard: MutexGuard<'a, WorkerStatus<'tcx>>,
    condvar: &'a Condvar,
}

impl<'tcx> WorkerParkingLot<'tcx> {
    const fn new() -> Self {
        Self { condvar: Condvar::new(), status: Mutex::new(WorkerStatus::Free) }
    }

    fn park<'a>(
        &self,
        validate: impl FnOnce() -> Option<QueryWaiter<'a, 'tcx>>,
        jobserver_proxy: &jobserver::Proxy,
    ) -> Result<(), Cycle<'tcx>>
    where
        'tcx: 'a,
    {
        let mut status_lock = self.status.lock();
        assert!(
            matches!(*status_lock, WorkerStatus::Free),
            "tried to park on a used worker parking lot"
        );
        let Some(waiter) = validate() else {
            return Ok(());
        };
        rustc_thread_pool::mark_blocked();
        jobserver_proxy.release_thread();
        // SAFETY: WorkerParkingLot::lock_waiting makes sure transmuted lifetime remains valid
        unsafe {
            *status_lock = WorkerStatus::Waiting(mem::transmute(waiter));
        }
        self.condvar.wait(&mut status_lock);
        // Spurious wakes aren't possible in parking_lot
        let res = match mem::replace(&mut *status_lock, WorkerStatus::Free) {
            WorkerStatus::Free => Ok(()),
            WorkerStatus::Cycle(cycle) => Err(cycle),
            WorkerStatus::Waiting(waiter) => {
                span_bug!(waiter.span, "unexpectedly found a waiter after unparking")
            }
        };
        drop(status_lock);
        jobserver_proxy.acquire_thread();
        res
    }

    #[inline]
    fn lock_waiter(&self) -> Option<QueryWaiterGuard<'_, 'tcx>> {
        let status_lock = self.status.lock();
        match *status_lock {
            WorkerStatus::Free => None,
            WorkerStatus::Cycle(_) => {
                panic!("Waiter thread unexpectedly has a pending cycle error")
            }
            WorkerStatus::Waiting(_) => {
                Some(QueryWaiterGuard { guard: status_lock, condvar: &self.condvar })
            }
        }
    }
}

impl<'a, 'tcx> QueryWaiterGuard<'a, 'tcx> {
    #[inline]
    pub fn span(&self) -> Span {
        let WorkerStatus::Waiting(waiter) = &*self.guard else { panic!() };
        waiter.span
    }

    #[inline]
    pub fn parent(&self) -> Option<QueryJobRef<'_, 'tcx>> {
        let WorkerStatus::Waiting(waiter) = &*self.guard else { panic!() };
        waiter.parent
    }

    #[inline]
    fn unpark(mut self) {
        debug_assert!(matches!(*self.guard, WorkerStatus::Waiting(_)));
        *self.guard = WorkerStatus::Free;
        drop(self.guard);
        assert!(self.condvar.notify_one());
    }

    #[inline]
    pub fn unpark_with_cycle(mut self, cycle: Cycle<'tcx>, registry: &Registry) {
        debug_assert!(matches!(*self.guard, WorkerStatus::Waiting(_)));
        *self.guard = WorkerStatus::Cycle(cycle);
        rustc_thread_pool::mark_unblocked(registry, 1);
        drop(self.guard);
        assert!(self.condvar.notify_one());
    }
}
