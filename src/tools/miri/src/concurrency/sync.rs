use std::cell::RefCell;
use std::collections::VecDeque;
use std::collections::hash_map::Entry;
use std::default::Default;
use std::ops::Not;
use std::rc::Rc;
use std::time::Duration;

use rustc_abi::Size;
use rustc_data_structures::fx::FxHashMap;
use rustc_index::{Idx, IndexVec};

use super::init_once::InitOnce;
use super::vector_clock::VClock;
use crate::*;

/// We cannot use the `newtype_index!` macro because we have to use 0 as a
/// sentinel value meaning that the identifier is not assigned. This is because
/// the pthreads static initializers initialize memory with zeros (see the
/// `src/shims/sync.rs` file).
macro_rules! declare_id {
    ($name: ident) => {
        /// 0 is used to indicate that the id was not yet assigned and,
        /// therefore, is not a valid identifier.
        #[derive(Clone, Copy, Debug, PartialOrd, Ord, PartialEq, Eq, Hash)]
        pub struct $name(std::num::NonZero<u32>);

        impl $crate::VisitProvenance for $name {
            fn visit_provenance(&self, _visit: &mut VisitWith<'_>) {}
        }

        impl Idx for $name {
            fn new(idx: usize) -> Self {
                // We use 0 as a sentinel value (see the comment above) and,
                // therefore, need to shift by one when converting from an index
                // into a vector.
                let shifted_idx = u32::try_from(idx).unwrap().strict_add(1);
                $name(std::num::NonZero::new(shifted_idx).unwrap())
            }
            fn index(self) -> usize {
                // See the comment in `Self::new`.
                // (This cannot underflow because `self.0` is `NonZero<u32>`.)
                usize::try_from(self.0.get() - 1).unwrap()
            }
        }
    };
}
pub(super) use declare_id;

/// The mutex state.
#[derive(Default, Debug)]
struct Mutex {
    /// The thread that currently owns the lock.
    owner: Option<ThreadId>,
    /// How many times the mutex was locked by the owner.
    lock_count: usize,
    /// The queue of threads waiting for this mutex.
    queue: VecDeque<ThreadId>,
    /// Mutex clock. This tracks the moment of the last unlock.
    clock: VClock,
}

#[derive(Default, Clone, Debug)]
pub struct MutexRef(Rc<RefCell<Mutex>>);

impl MutexRef {
    fn new() -> Self {
        MutexRef(Rc::new(RefCell::new(Mutex::default())))
    }

    /// Get the id of the thread that currently owns this lock, or `None` if it is not locked.
    pub fn owner(&self) -> Option<ThreadId> {
        self.0.borrow().owner
    }
}

impl VisitProvenance for MutexRef {
    fn visit_provenance(&self, _visit: &mut VisitWith<'_>) {
        // Mutex contains no provenance.
    }
}

/// The read-write lock state.
#[derive(Default, Debug)]
struct RwLock {
    /// The writer thread that currently owns the lock.
    writer: Option<ThreadId>,
    /// The readers that currently own the lock and how many times they acquired
    /// the lock.
    readers: FxHashMap<ThreadId, usize>,
    /// The queue of writer threads waiting for this lock.
    writer_queue: VecDeque<ThreadId>,
    /// The queue of reader threads waiting for this lock.
    reader_queue: VecDeque<ThreadId>,
    /// Data race clock for writers. Tracks the happens-before
    /// ordering between each write access to a rwlock and is updated
    /// after a sequence of concurrent readers to track the happens-
    /// before ordering between the set of previous readers and
    /// the current writer.
    /// Contains the clock of the last thread to release a writer
    /// lock or the joined clock of the set of last threads to release
    /// shared reader locks.
    clock_unlocked: VClock,
    /// Data race clock for readers. This is temporary storage
    /// for the combined happens-before ordering for between all
    /// concurrent readers and the next writer, and the value
    /// is stored to the main data_race variable once all
    /// readers are finished.
    /// Has to be stored separately since reader lock acquires
    /// must load the clock of the last write and must not
    /// add happens-before orderings between shared reader
    /// locks.
    /// This is only relevant when there is an active reader.
    clock_current_readers: VClock,
}

impl RwLock {
    #[inline]
    /// Check if locked.
    fn is_locked(&self) -> bool {
        trace!(
            "rwlock_is_locked: writer is {:?} and there are {} reader threads (some of which could hold multiple read locks)",
            self.writer,
            self.readers.len(),
        );
        self.writer.is_some() || self.readers.is_empty().not()
    }

    /// Check if write locked.
    #[inline]
    fn is_write_locked(&self) -> bool {
        trace!("rwlock_is_write_locked: writer is {:?}", self.writer);
        self.writer.is_some()
    }
}

#[derive(Default, Clone, Debug)]
pub struct RwLockRef(Rc<RefCell<RwLock>>);

impl RwLockRef {
    fn new() -> Self {
        RwLockRef(Rc::new(RefCell::new(RwLock::default())))
    }

    pub fn is_locked(&self) -> bool {
        self.0.borrow().is_locked()
    }

    pub fn is_write_locked(&self) -> bool {
        self.0.borrow().is_write_locked()
    }
}

impl VisitProvenance for RwLockRef {
    fn visit_provenance(&self, _visit: &mut VisitWith<'_>) {
        // RwLockRef contains no provenance.
    }
}

declare_id!(CondvarId);

/// The conditional variable state.
#[derive(Default, Debug)]
struct Condvar {
    waiters: VecDeque<ThreadId>,
    /// Tracks the happens-before relationship
    /// between a cond-var signal and a cond-var
    /// wait during a non-spurious signal event.
    /// Contains the clock of the last thread to
    /// perform a condvar-signal.
    clock: VClock,
}

/// The futex state.
#[derive(Default, Debug)]
struct Futex {
    waiters: Vec<FutexWaiter>,
    /// Tracks the happens-before relationship
    /// between a futex-wake and a futex-wait
    /// during a non-spurious wake event.
    /// Contains the clock of the last thread to
    /// perform a futex-wake.
    clock: VClock,
}

#[derive(Default, Clone)]
pub struct FutexRef(Rc<RefCell<Futex>>);

impl FutexRef {
    pub fn waiters(&self) -> usize {
        self.0.borrow().waiters.len()
    }
}

impl VisitProvenance for FutexRef {
    fn visit_provenance(&self, _visit: &mut VisitWith<'_>) {
        // No provenance in `Futex`.
    }
}

/// A thread waiting on a futex.
#[derive(Debug)]
struct FutexWaiter {
    /// The thread that is waiting on this futex.
    thread: ThreadId,
    /// The bitset used by FUTEX_*_BITSET, or u32::MAX for other operations.
    bitset: u32,
}

/// The state of all synchronization objects.
#[derive(Default, Debug)]
pub struct SynchronizationObjects {
    condvars: IndexVec<CondvarId, Condvar>,
    pub(super) init_onces: IndexVec<InitOnceId, InitOnce>,
}

// Private extension trait for local helper methods
impl<'tcx> EvalContextExtPriv<'tcx> for crate::MiriInterpCx<'tcx> {}
pub(super) trait EvalContextExtPriv<'tcx>: crate::MiriInterpCxExt<'tcx> {
    fn condvar_reacquire_mutex(
        &mut self,
        mutex_ref: MutexRef,
        retval: Scalar,
        dest: MPlaceTy<'tcx>,
    ) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();
        if let Some(owner) = mutex_ref.owner() {
            assert_ne!(owner, this.active_thread());
            this.mutex_enqueue_and_block(mutex_ref, Some((retval, dest)));
        } else {
            // We can have it right now!
            this.mutex_lock(&mutex_ref);
            // Don't forget to write the return value.
            this.write_scalar(retval, &dest)?;
        }
        interp_ok(())
    }
}

impl SynchronizationObjects {
    pub fn mutex_create(&mut self) -> MutexRef {
        MutexRef::new()
    }
    pub fn rwlock_create(&mut self) -> RwLockRef {
        RwLockRef::new()
    }

    pub fn condvar_create(&mut self) -> CondvarId {
        self.condvars.push(Default::default())
    }

    pub fn init_once_create(&mut self) -> InitOnceId {
        self.init_onces.push(Default::default())
    }
}

impl<'tcx> AllocExtra<'tcx> {
    fn get_sync<T: 'static>(&self, offset: Size) -> Option<&T> {
        self.sync.get(&offset).and_then(|s| s.downcast_ref::<T>())
    }
}

/// We designate an `init`` field in all primitives.
/// If `init` is set to this, we consider the primitive initialized.
pub const LAZY_INIT_COOKIE: u32 = 0xcafe_affe;

// Public interface to synchronization primitives. Please note that in most
// cases, the function calls are infallible and it is the client's (shim
// implementation's) responsibility to detect and deal with erroneous
// situations.
impl<'tcx> EvalContextExt<'tcx> for crate::MiriInterpCx<'tcx> {}
pub trait EvalContextExt<'tcx>: crate::MiriInterpCxExt<'tcx> {
    /// Helper for lazily initialized `alloc_extra.sync` data:
    /// this forces an immediate init.
    /// Return a reference to the data in the machine state.
    fn lazy_sync_init<'a, T: 'static>(
        &'a mut self,
        primitive: &MPlaceTy<'tcx>,
        init_offset: Size,
        data: T,
    ) -> InterpResult<'tcx, &'a T>
    where
        'tcx: 'a,
    {
        let this = self.eval_context_mut();

        let (alloc, offset, _) = this.ptr_get_alloc_id(primitive.ptr(), 0)?;
        let (alloc_extra, _machine) = this.get_alloc_extra_mut(alloc)?;
        alloc_extra.sync.insert(offset, Box::new(data));
        // Mark this as "initialized".
        let init_field = primitive.offset(init_offset, this.machine.layouts.u32, this)?;
        this.write_scalar_atomic(
            Scalar::from_u32(LAZY_INIT_COOKIE),
            &init_field,
            AtomicWriteOrd::Relaxed,
        )?;
        interp_ok(this.get_alloc_extra(alloc)?.get_sync::<T>(offset).unwrap())
    }

    /// Helper for lazily initialized `alloc_extra.sync` data:
    /// Checks if the primitive is initialized:
    /// - If yes, fetches the data from `alloc_extra.sync`, or calls `missing_data` if that fails
    ///   and stores that in `alloc_extra.sync`.
    /// - Otherwise, calls `new_data` to initialize the primitive.
    ///
    /// Return a reference to the data in the machine state.
    fn lazy_sync_get_data<'a, T: 'static>(
        &'a mut self,
        primitive: &MPlaceTy<'tcx>,
        init_offset: Size,
        missing_data: impl FnOnce() -> InterpResult<'tcx, T>,
        new_data: impl FnOnce(&mut MiriInterpCx<'tcx>) -> InterpResult<'tcx, T>,
    ) -> InterpResult<'tcx, &'a T>
    where
        'tcx: 'a,
    {
        let this = self.eval_context_mut();

        // Check if this is already initialized. Needs to be atomic because we can race with another
        // thread initializing. Needs to be an RMW operation to ensure we read the *latest* value.
        // So we just try to replace MUTEX_INIT_COOKIE with itself.
        let init_cookie = Scalar::from_u32(LAZY_INIT_COOKIE);
        let init_field = primitive.offset(init_offset, this.machine.layouts.u32, this)?;
        let (_init, success) = this
            .atomic_compare_exchange_scalar(
                &init_field,
                &ImmTy::from_scalar(init_cookie, this.machine.layouts.u32),
                init_cookie,
                AtomicRwOrd::Relaxed,
                AtomicReadOrd::Relaxed,
                /* can_fail_spuriously */ false,
            )?
            .to_scalar_pair();

        if success.to_bool()? {
            // If it is initialized, it must be found in the "sync primitive" table,
            // or else it has been moved illegally.
            let (alloc, offset, _) = this.ptr_get_alloc_id(primitive.ptr(), 0)?;
            let (alloc_extra, _machine) = this.get_alloc_extra_mut(alloc)?;
            // Due to borrow checker reasons, we have to do the lookup twice.
            if alloc_extra.get_sync::<T>(offset).is_none() {
                let data = missing_data()?;
                alloc_extra.sync.insert(offset, Box::new(data));
            }
            interp_ok(alloc_extra.get_sync::<T>(offset).unwrap())
        } else {
            let data = new_data(this)?;
            this.lazy_sync_init(primitive, init_offset, data)
        }
    }

    /// Get the synchronization primitive associated with the given pointer,
    /// or initialize a new one.
    ///
    /// Return `None` if this pointer does not point to at least 1 byte of mutable memory.
    fn get_sync_or_init<'a, T: 'static>(
        &'a mut self,
        ptr: Pointer,
        new: impl FnOnce(&'a mut MiriMachine<'tcx>) -> T,
    ) -> Option<&'a T>
    where
        'tcx: 'a,
    {
        let this = self.eval_context_mut();
        if !this.ptr_try_get_alloc_id(ptr, 0).ok().is_some_and(|(alloc_id, offset, ..)| {
            let info = this.get_alloc_info(alloc_id);
            info.kind == AllocKind::LiveData && info.mutbl.is_mut() && offset < info.size
        }) {
            return None;
        }
        // This cannot fail now.
        let (alloc, offset, _) = this.ptr_get_alloc_id(ptr, 0).unwrap();
        let (alloc_extra, machine) = this.get_alloc_extra_mut(alloc).unwrap();
        // Due to borrow checker reasons, we have to do the lookup twice.
        if alloc_extra.get_sync::<T>(offset).is_none() {
            let new = new(machine);
            alloc_extra.sync.insert(offset, Box::new(new));
        }
        Some(alloc_extra.get_sync::<T>(offset).unwrap())
    }

    /// Lock by setting the mutex owner and increasing the lock count.
    fn mutex_lock(&mut self, mutex_ref: &MutexRef) {
        let this = self.eval_context_mut();
        let thread = this.active_thread();
        let mut mutex = mutex_ref.0.borrow_mut();
        if let Some(current_owner) = mutex.owner {
            assert_eq!(thread, current_owner, "mutex already locked by another thread");
            assert!(
                mutex.lock_count > 0,
                "invariant violation: lock_count == 0 iff the thread is unlocked"
            );
        } else {
            mutex.owner = Some(thread);
        }
        mutex.lock_count = mutex.lock_count.strict_add(1);
        if let Some(data_race) = this.machine.data_race.as_vclocks_ref() {
            data_race.acquire_clock(&mutex.clock, &this.machine.threads);
        }
    }

    /// Try unlocking by decreasing the lock count and returning the old lock
    /// count. If the lock count reaches 0, release the lock and potentially
    /// give to a new owner. If the lock was not locked by the current thread,
    /// return `None`.
    fn mutex_unlock(&mut self, mutex_ref: &MutexRef) -> InterpResult<'tcx, Option<usize>> {
        let this = self.eval_context_mut();
        let mut mutex = mutex_ref.0.borrow_mut();
        interp_ok(if let Some(current_owner) = mutex.owner {
            // Mutex is locked.
            if current_owner != this.machine.threads.active_thread() {
                // Only the owner can unlock the mutex.
                return interp_ok(None);
            }
            let old_lock_count = mutex.lock_count;
            mutex.lock_count = old_lock_count.strict_sub(1);
            if mutex.lock_count == 0 {
                mutex.owner = None;
                // The mutex is completely unlocked. Try transferring ownership
                // to another thread.

                if let Some(data_race) = this.machine.data_race.as_vclocks_ref() {
                    data_race.release_clock(&this.machine.threads, |clock| {
                        mutex.clock.clone_from(clock)
                    });
                }
                let thread_id = mutex.queue.pop_front();
                // We need to drop our mutex borrow before unblock_thread
                // because it will be borrowed again in the unblock callback.
                drop(mutex);
                if thread_id.is_some() {
                    this.unblock_thread(thread_id.unwrap(), BlockReason::Mutex)?;
                }
            }
            Some(old_lock_count)
        } else {
            // Mutex is not locked.
            None
        })
    }

    /// Put the thread into the queue waiting for the mutex.
    ///
    /// Once the Mutex becomes available and if it exists, `retval_dest.0` will
    /// be written to `retval_dest.1`.
    #[inline]
    fn mutex_enqueue_and_block(
        &mut self,
        mutex_ref: MutexRef,
        retval_dest: Option<(Scalar, MPlaceTy<'tcx>)>,
    ) {
        let this = self.eval_context_mut();
        let thread = this.active_thread();
        let mut mutex = mutex_ref.0.borrow_mut();
        mutex.queue.push_back(thread);
        assert!(mutex.owner.is_some(), "queuing on unlocked mutex");
        drop(mutex);
        this.block_thread(
            BlockReason::Mutex,
            None,
            callback!(
                @capture<'tcx> {
                    mutex_ref: MutexRef,
                    retval_dest: Option<(Scalar, MPlaceTy<'tcx>)>,
                }
                |this, unblock: UnblockKind| {
                    assert_eq!(unblock, UnblockKind::Ready);

                    assert!(mutex_ref.owner().is_none());
                    this.mutex_lock(&mutex_ref);

                    if let Some((retval, dest)) = retval_dest {
                        this.write_scalar(retval, &dest)?;
                    }

                    interp_ok(())
                }
            ),
        );
    }

    /// Read-lock the lock by adding the `reader` the list of threads that own
    /// this lock.
    fn rwlock_reader_lock(&mut self, rwlock_ref: &RwLockRef) {
        let this = self.eval_context_mut();
        let thread = this.active_thread();
        trace!("rwlock_reader_lock: now also held (one more time) by {:?}", thread);
        let mut rwlock = rwlock_ref.0.borrow_mut();
        assert!(!rwlock.is_write_locked(), "the lock is write locked");
        let count = rwlock.readers.entry(thread).or_insert(0);
        *count = count.strict_add(1);
        if let Some(data_race) = this.machine.data_race.as_vclocks_ref() {
            data_race.acquire_clock(&rwlock.clock_unlocked, &this.machine.threads);
        }
    }

    /// Try read-unlock the lock for the current threads and potentially give the lock to a new owner.
    /// Returns `true` if succeeded, `false` if this `reader` did not hold the lock.
    fn rwlock_reader_unlock(&mut self, rwlock_ref: &RwLockRef) -> InterpResult<'tcx, bool> {
        let this = self.eval_context_mut();
        let thread = this.active_thread();
        let mut rwlock = rwlock_ref.0.borrow_mut();
        match rwlock.readers.entry(thread) {
            Entry::Occupied(mut entry) => {
                let count = entry.get_mut();
                assert!(*count > 0, "rwlock locked with count == 0");
                *count -= 1;
                if *count == 0 {
                    trace!("rwlock_reader_unlock: no longer held by {:?}", thread);
                    entry.remove();
                } else {
                    trace!("rwlock_reader_unlock: held one less time by {:?}", thread);
                }
            }
            Entry::Vacant(_) => return interp_ok(false), // we did not even own this lock
        }
        if let Some(data_race) = this.machine.data_race.as_vclocks_ref() {
            // Add this to the shared-release clock of all concurrent readers.
            data_race.release_clock(&this.machine.threads, |clock| {
                rwlock.clock_current_readers.join(clock)
            });
        }

        // The thread was a reader. If the lock is not held any more, give it to a writer.
        if rwlock.is_locked().not() {
            // All the readers are finished, so set the writer data-race handle to the value
            // of the union of all reader data race handles, since the set of readers
            // happen-before the writers
            let rwlock_ref = &mut *rwlock;
            rwlock_ref.clock_unlocked.clone_from(&rwlock_ref.clock_current_readers);
            // See if there is a thread to unblock.
            if let Some(writer) = rwlock_ref.writer_queue.pop_front() {
                drop(rwlock); // make RefCell available for unblock callback
                this.unblock_thread(writer, BlockReason::RwLock)?;
            }
        }
        interp_ok(true)
    }

    /// Put the reader in the queue waiting for the lock and block it.
    /// Once the lock becomes available, `retval` will be written to `dest`.
    #[inline]
    fn rwlock_enqueue_and_block_reader(
        &mut self,
        rwlock_ref: RwLockRef,
        retval: Scalar,
        dest: MPlaceTy<'tcx>,
    ) {
        let this = self.eval_context_mut();
        let thread = this.active_thread();
        let mut rwlock = rwlock_ref.0.borrow_mut();
        rwlock.reader_queue.push_back(thread);
        assert!(rwlock.is_write_locked(), "read-queueing on not write locked rwlock");
        drop(rwlock);
        this.block_thread(
            BlockReason::RwLock,
            None,
            callback!(
                @capture<'tcx> {
                    rwlock_ref: RwLockRef,
                    retval: Scalar,
                    dest: MPlaceTy<'tcx>,
                }
                |this, unblock: UnblockKind| {
                    assert_eq!(unblock, UnblockKind::Ready);
                    this.rwlock_reader_lock(&rwlock_ref);
                    this.write_scalar(retval, &dest)?;
                    interp_ok(())
                }
            ),
        );
    }

    /// Lock by setting the writer that owns the lock.
    #[inline]
    fn rwlock_writer_lock(&mut self, rwlock_ref: &RwLockRef) {
        let this = self.eval_context_mut();
        let thread = this.active_thread();
        trace!("rwlock_writer_lock: now held by {:?}", thread);

        let mut rwlock = rwlock_ref.0.borrow_mut();
        assert!(!rwlock.is_locked(), "the rwlock is already locked");
        rwlock.writer = Some(thread);
        if let Some(data_race) = this.machine.data_race.as_vclocks_ref() {
            data_race.acquire_clock(&rwlock.clock_unlocked, &this.machine.threads);
        }
    }

    /// Try to unlock an rwlock held by the current thread.
    /// Return `false` if it is held by another thread.
    #[inline]
    fn rwlock_writer_unlock(&mut self, rwlock_ref: &RwLockRef) -> InterpResult<'tcx, bool> {
        let this = self.eval_context_mut();
        let thread = this.active_thread();
        let mut rwlock = rwlock_ref.0.borrow_mut();
        interp_ok(if let Some(current_writer) = rwlock.writer {
            if current_writer != thread {
                // Only the owner can unlock the rwlock.
                return interp_ok(false);
            }
            rwlock.writer = None;
            trace!("rwlock_writer_unlock: unlocked by {:?}", thread);
            // Record release clock for next lock holder.
            if let Some(data_race) = this.machine.data_race.as_vclocks_ref() {
                data_race.release_clock(&this.machine.threads, |clock| {
                    rwlock.clock_unlocked.clone_from(clock)
                });
            }

            // The thread was a writer.
            //
            // We are prioritizing writers here against the readers. As a
            // result, not only readers can starve writers, but also writers can
            // starve readers.
            if let Some(writer) = rwlock.writer_queue.pop_front() {
                drop(rwlock); // make RefCell available for unblock callback
                this.unblock_thread(writer, BlockReason::RwLock)?;
            } else {
                // Take the entire read queue and wake them all up.
                let readers = std::mem::take(&mut rwlock.reader_queue);
                drop(rwlock); // make RefCell available for unblock callback
                for reader in readers {
                    this.unblock_thread(reader, BlockReason::RwLock)?;
                }
            }
            true
        } else {
            false
        })
    }

    /// Put the writer in the queue waiting for the lock.
    /// Once the lock becomes available, `retval` will be written to `dest`.
    #[inline]
    fn rwlock_enqueue_and_block_writer(
        &mut self,
        rwlock_ref: RwLockRef,
        retval: Scalar,
        dest: MPlaceTy<'tcx>,
    ) {
        let this = self.eval_context_mut();
        let thread = this.active_thread();
        let mut rwlock = rwlock_ref.0.borrow_mut();
        rwlock.writer_queue.push_back(thread);
        assert!(rwlock.is_locked(), "write-queueing on unlocked rwlock");
        drop(rwlock);
        this.block_thread(
            BlockReason::RwLock,
            None,
            callback!(
                @capture<'tcx> {
                    rwlock_ref: RwLockRef,
                    retval: Scalar,
                    dest: MPlaceTy<'tcx>,
                }
                |this, unblock: UnblockKind| {
                    assert_eq!(unblock, UnblockKind::Ready);
                    this.rwlock_writer_lock(&rwlock_ref);
                    this.write_scalar(retval, &dest)?;
                    interp_ok(())
                }
            ),
        );
    }

    /// Is the conditional variable awaited?
    #[inline]
    fn condvar_is_awaited(&mut self, id: CondvarId) -> bool {
        let this = self.eval_context_mut();
        !this.machine.sync.condvars[id].waiters.is_empty()
    }

    /// Release the mutex and let the current thread wait on the given condition variable.
    /// Once it is signaled, the mutex will be acquired and `retval_succ` will be written to `dest`.
    /// If the timeout happens first, `retval_timeout` will be written to `dest`.
    fn condvar_wait(
        &mut self,
        condvar: CondvarId,
        mutex_ref: MutexRef,
        timeout: Option<(TimeoutClock, TimeoutAnchor, Duration)>,
        retval_succ: Scalar,
        retval_timeout: Scalar,
        dest: MPlaceTy<'tcx>,
    ) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();
        if let Some(old_locked_count) = this.mutex_unlock(&mutex_ref)? {
            if old_locked_count != 1 {
                throw_unsup_format!(
                    "awaiting a condvar on a mutex acquired multiple times is not supported"
                );
            }
        } else {
            throw_ub_format!(
                "awaiting a condvar on a mutex that is unlocked or owned by a different thread"
            );
        }
        let thread = this.active_thread();
        let waiters = &mut this.machine.sync.condvars[condvar].waiters;
        waiters.push_back(thread);
        this.block_thread(
            BlockReason::Condvar(condvar),
            timeout,
            callback!(
                @capture<'tcx> {
                    condvar: CondvarId,
                    mutex_ref: MutexRef,
                    retval_succ: Scalar,
                    retval_timeout: Scalar,
                    dest: MPlaceTy<'tcx>,
                }
                |this, unblock: UnblockKind| {
                    match unblock {
                        UnblockKind::Ready => {
                            // The condvar was signaled. Make sure we get the clock for that.
                            if let Some(data_race) = this.machine.data_race.as_vclocks_ref() {
                                data_race.acquire_clock(
                                    &this.machine.sync.condvars[condvar].clock,
                                    &this.machine.threads,
                                );
                            }
                            // Try to acquire the mutex.
                            // The timeout only applies to the first wait (until the signal), not for mutex acquisition.
                            this.condvar_reacquire_mutex(mutex_ref, retval_succ, dest)
                        }
                        UnblockKind::TimedOut => {
                            // We have to remove the waiter from the queue again.
                            let thread = this.active_thread();
                            let waiters = &mut this.machine.sync.condvars[condvar].waiters;
                            waiters.retain(|waiter| *waiter != thread);
                            // Now get back the lock.
                            this.condvar_reacquire_mutex(mutex_ref, retval_timeout, dest)
                        }
                    }
                }
            ),
        );
        interp_ok(())
    }

    /// Wake up some thread (if there is any) sleeping on the conditional
    /// variable. Returns `true` iff any thread was woken up.
    fn condvar_signal(&mut self, id: CondvarId) -> InterpResult<'tcx, bool> {
        let this = self.eval_context_mut();
        let condvar = &mut this.machine.sync.condvars[id];

        // Each condvar signal happens-before the end of the condvar wake
        if let Some(data_race) = this.machine.data_race.as_vclocks_ref() {
            data_race.release_clock(&this.machine.threads, |clock| condvar.clock.clone_from(clock));
        }
        let Some(waiter) = condvar.waiters.pop_front() else {
            return interp_ok(false);
        };
        this.unblock_thread(waiter, BlockReason::Condvar(id))?;
        interp_ok(true)
    }

    /// Wait for the futex to be signaled, or a timeout. Once the thread is
    /// unblocked, `callback` is called with the unblock reason.
    fn futex_wait(
        &mut self,
        futex_ref: FutexRef,
        bitset: u32,
        timeout: Option<(TimeoutClock, TimeoutAnchor, Duration)>,
        callback: DynUnblockCallback<'tcx>,
    ) {
        let this = self.eval_context_mut();
        let thread = this.active_thread();
        let mut futex = futex_ref.0.borrow_mut();
        let waiters = &mut futex.waiters;
        assert!(waiters.iter().all(|waiter| waiter.thread != thread), "thread is already waiting");
        waiters.push(FutexWaiter { thread, bitset });
        drop(futex);

        this.block_thread(
            BlockReason::Futex,
            timeout,
            callback!(
                @capture<'tcx> {
                    futex_ref: FutexRef,
                    callback: DynUnblockCallback<'tcx>,
                }
                |this, unblock: UnblockKind| {
                    match unblock {
                        UnblockKind::Ready => {
                            let futex = futex_ref.0.borrow();
                            // Acquire the clock of the futex.
                            if let Some(data_race) = this.machine.data_race.as_vclocks_ref() {
                                data_race.acquire_clock(&futex.clock, &this.machine.threads);
                            }
                        },
                        UnblockKind::TimedOut => {
                            // Remove the waiter from the futex.
                            let thread = this.active_thread();
                            let mut futex = futex_ref.0.borrow_mut();
                            futex.waiters.retain(|waiter| waiter.thread != thread);
                        },
                    }

                    callback.call(this, unblock)
                }
            ),
        );
    }

    /// Wake up `count` of the threads in the queue that match any of the bits
    /// in the bitset. Returns how many threads were woken.
    fn futex_wake(
        &mut self,
        futex_ref: &FutexRef,
        bitset: u32,
        count: usize,
    ) -> InterpResult<'tcx, usize> {
        let this = self.eval_context_mut();
        let mut futex = futex_ref.0.borrow_mut();

        // Each futex-wake happens-before the end of the futex wait
        if let Some(data_race) = this.machine.data_race.as_vclocks_ref() {
            data_race.release_clock(&this.machine.threads, |clock| futex.clock.clone_from(clock));
        }

        // Remove `count` of the threads in the queue that match any of the bits in the bitset.
        // We collect all of them before unblocking because the unblock callback may access the
        // futex state to retrieve the remaining number of waiters on macOS.
        let waiters: Vec<_> =
            futex.waiters.extract_if(.., |w| w.bitset & bitset != 0).take(count).collect();
        drop(futex);

        let woken = waiters.len();
        for waiter in waiters {
            this.unblock_thread(waiter.thread, BlockReason::Futex)?;
        }

        interp_ok(woken)
    }
}
