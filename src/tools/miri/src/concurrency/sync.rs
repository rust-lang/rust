use std::collections::{hash_map::Entry, VecDeque};
use std::num::NonZeroU32;
use std::ops::Not;

use log::trace;

use rustc_data_structures::fx::FxHashMap;
use rustc_index::{Idx, IndexVec};
use rustc_middle::ty::layout::TyAndLayout;

use super::init_once::InitOnce;
use super::vector_clock::VClock;
use crate::*;

pub trait SyncId {
    fn from_u32(id: u32) -> Self;
    fn to_u32(&self) -> u32;
}

/// We cannot use the `newtype_index!` macro because we have to use 0 as a
/// sentinel value meaning that the identifier is not assigned. This is because
/// the pthreads static initializers initialize memory with zeros (see the
/// `src/shims/sync.rs` file).
macro_rules! declare_id {
    ($name: ident) => {
        /// 0 is used to indicate that the id was not yet assigned and,
        /// therefore, is not a valid identifier.
        #[derive(Clone, Copy, Debug, PartialOrd, Ord, PartialEq, Eq, Hash)]
        pub struct $name(NonZeroU32);

        impl SyncId for $name {
            // Panics if `id == 0`.
            fn from_u32(id: u32) -> Self {
                Self(NonZeroU32::new(id).unwrap())
            }
            fn to_u32(&self) -> u32 {
                self.0.get()
            }
        }

        impl Idx for $name {
            fn new(idx: usize) -> Self {
                // We use 0 as a sentinel value (see the comment above) and,
                // therefore, need to shift by one when converting from an index
                // into a vector.
                let shifted_idx = u32::try_from(idx).unwrap().checked_add(1).unwrap();
                $name(NonZeroU32::new(shifted_idx).unwrap())
            }
            fn index(self) -> usize {
                // See the comment in `Self::new`.
                // (This cannot underflow because self is NonZeroU32.)
                usize::try_from(self.0.get() - 1).unwrap()
            }
        }

        impl $name {
            pub fn to_u32_scalar(&self) -> Scalar<Provenance> {
                Scalar::from_u32(self.0.get())
            }
        }
    };
}

declare_id!(MutexId);

/// The mutex state.
#[derive(Default, Debug)]
struct Mutex {
    /// The thread that currently owns the lock.
    owner: Option<ThreadId>,
    /// How many times the mutex was locked by the owner.
    lock_count: usize,
    /// The queue of threads waiting for this mutex.
    queue: VecDeque<ThreadId>,
    /// Data race handle, this tracks the happens-before
    /// relationship between each mutex access. It is
    /// released to during unlock and acquired from during
    /// locking, and therefore stores the clock of the last
    /// thread to release this mutex.
    data_race: VClock,
}

declare_id!(RwLockId);

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
    /// Data race handle for writers, tracks the happens-before
    /// ordering between each write access to a rwlock and is updated
    /// after a sequence of concurrent readers to track the happens-
    /// before ordering between the set of previous readers and
    /// the current writer.
    /// Contains the clock of the last thread to release a writer
    /// lock or the joined clock of the set of last threads to release
    /// shared reader locks.
    data_race: VClock,
    /// Data race handle for readers, this is temporary storage
    /// for the combined happens-before ordering for between all
    /// concurrent readers and the next writer, and the value
    /// is stored to the main data_race variable once all
    /// readers are finished.
    /// Has to be stored separately since reader lock acquires
    /// must load the clock of the last write and must not
    /// add happens-before orderings between shared reader
    /// locks.
    data_race_reader: VClock,
}

declare_id!(CondvarId);

#[derive(Debug, Copy, Clone)]
pub enum RwLockMode {
    Read,
    Write,
}

#[derive(Debug)]
pub enum CondvarLock {
    Mutex(MutexId),
    RwLock { id: RwLockId, mode: RwLockMode },
}

/// A thread waiting on a conditional variable.
#[derive(Debug)]
struct CondvarWaiter {
    /// The thread that is waiting on this variable.
    thread: ThreadId,
    /// The mutex or rwlock on which the thread is waiting.
    lock: CondvarLock,
}

/// The conditional variable state.
#[derive(Default, Debug)]
struct Condvar {
    waiters: VecDeque<CondvarWaiter>,
    /// Tracks the happens-before relationship
    /// between a cond-var signal and a cond-var
    /// wait during a non-spurious signal event.
    /// Contains the clock of the last thread to
    /// perform a futex-signal.
    data_race: VClock,
}

/// The futex state.
#[derive(Default, Debug)]
struct Futex {
    waiters: VecDeque<FutexWaiter>,
    /// Tracks the happens-before relationship
    /// between a futex-wake and a futex-wait
    /// during a non-spurious wake event.
    /// Contains the clock of the last thread to
    /// perform a futex-wake.
    data_race: VClock,
}

/// A thread waiting on a futex.
#[derive(Debug)]
struct FutexWaiter {
    /// The thread that is waiting on this futex.
    thread: ThreadId,
    /// The bitset used by FUTEX_*_BITSET, or u32::MAX for other operations.
    bitset: u32,
}

/// The state of all synchronization variables.
#[derive(Default, Debug)]
pub(crate) struct SynchronizationState<'mir, 'tcx> {
    mutexes: IndexVec<MutexId, Mutex>,
    rwlocks: IndexVec<RwLockId, RwLock>,
    condvars: IndexVec<CondvarId, Condvar>,
    futexes: FxHashMap<u64, Futex>,
    pub(super) init_onces: IndexVec<InitOnceId, InitOnce<'mir, 'tcx>>,
}

impl<'mir, 'tcx> VisitTags for SynchronizationState<'mir, 'tcx> {
    fn visit_tags(&self, visit: &mut dyn FnMut(BorTag)) {
        for init_once in self.init_onces.iter() {
            init_once.visit_tags(visit);
        }
    }
}

// Private extension trait for local helper methods
impl<'mir, 'tcx: 'mir> EvalContextExtPriv<'mir, 'tcx> for crate::MiriInterpCx<'mir, 'tcx> {}
pub(super) trait EvalContextExtPriv<'mir, 'tcx: 'mir>:
    crate::MiriInterpCxExt<'mir, 'tcx>
{
    /// Lazily initialize the ID of this Miri sync structure.
    /// ('0' indicates uninit.)
    #[inline]
    fn get_or_create_id<Id: SyncId>(
        &mut self,
        next_id: Id,
        lock_op: &OpTy<'tcx, Provenance>,
        lock_layout: TyAndLayout<'tcx>,
        offset: u64,
    ) -> InterpResult<'tcx, Option<Id>> {
        let this = self.eval_context_mut();
        let value_place =
            this.deref_operand_and_offset(lock_op, offset, lock_layout, this.machine.layouts.u32)?;

        // Since we are lazy, this update has to be atomic.
        let (old, success) = this
            .atomic_compare_exchange_scalar(
                &value_place,
                &ImmTy::from_uint(0u32, this.machine.layouts.u32),
                Scalar::from_u32(next_id.to_u32()),
                AtomicRwOrd::Relaxed, // deliberately *no* synchronization
                AtomicReadOrd::Relaxed,
                false,
            )?
            .to_scalar_pair();

        Ok(if success.to_bool().expect("compare_exchange's second return value is a bool") {
            // Caller of the closure needs to allocate next_id
            None
        } else {
            Some(Id::from_u32(old.to_u32().expect("layout is u32")))
        })
    }

    /// Take a reader out of the queue waiting for the lock.
    /// Returns `true` if some thread got the rwlock.
    #[inline]
    fn rwlock_dequeue_and_lock_reader(&mut self, id: RwLockId) -> bool {
        let this = self.eval_context_mut();
        if let Some(reader) = this.machine.threads.sync.rwlocks[id].reader_queue.pop_front() {
            this.unblock_thread(reader);
            this.rwlock_reader_lock(id, reader);
            true
        } else {
            false
        }
    }

    /// Take the writer out of the queue waiting for the lock.
    /// Returns `true` if some thread got the rwlock.
    #[inline]
    fn rwlock_dequeue_and_lock_writer(&mut self, id: RwLockId) -> bool {
        let this = self.eval_context_mut();
        if let Some(writer) = this.machine.threads.sync.rwlocks[id].writer_queue.pop_front() {
            this.unblock_thread(writer);
            this.rwlock_writer_lock(id, writer);
            true
        } else {
            false
        }
    }

    /// Take a thread out of the queue waiting for the mutex, and lock
    /// the mutex for it. Returns `true` if some thread has the mutex now.
    #[inline]
    fn mutex_dequeue_and_lock(&mut self, id: MutexId) -> bool {
        let this = self.eval_context_mut();
        if let Some(thread) = this.machine.threads.sync.mutexes[id].queue.pop_front() {
            this.unblock_thread(thread);
            this.mutex_lock(id, thread);
            true
        } else {
            false
        }
    }
}

// Public interface to synchronization primitives. Please note that in most
// cases, the function calls are infallible and it is the client's (shim
// implementation's) responsibility to detect and deal with erroneous
// situations.
impl<'mir, 'tcx: 'mir> EvalContextExt<'mir, 'tcx> for crate::MiriInterpCx<'mir, 'tcx> {}
pub trait EvalContextExt<'mir, 'tcx: 'mir>: crate::MiriInterpCxExt<'mir, 'tcx> {
    fn mutex_get_or_create_id(
        &mut self,
        lock_op: &OpTy<'tcx, Provenance>,
        lock_layout: TyAndLayout<'tcx>,
        offset: u64,
    ) -> InterpResult<'tcx, MutexId> {
        let this = self.eval_context_mut();
        this.mutex_get_or_create(|ecx, next_id| {
            ecx.get_or_create_id(next_id, lock_op, lock_layout, offset)
        })
    }

    fn rwlock_get_or_create_id(
        &mut self,
        lock_op: &OpTy<'tcx, Provenance>,
        lock_layout: TyAndLayout<'tcx>,
        offset: u64,
    ) -> InterpResult<'tcx, RwLockId> {
        let this = self.eval_context_mut();
        this.rwlock_get_or_create(|ecx, next_id| {
            ecx.get_or_create_id(next_id, lock_op, lock_layout, offset)
        })
    }

    fn condvar_get_or_create_id(
        &mut self,
        lock_op: &OpTy<'tcx, Provenance>,
        lock_layout: TyAndLayout<'tcx>,
        offset: u64,
    ) -> InterpResult<'tcx, CondvarId> {
        let this = self.eval_context_mut();
        this.condvar_get_or_create(|ecx, next_id| {
            ecx.get_or_create_id(next_id, lock_op, lock_layout, offset)
        })
    }

    #[inline]
    /// Provides the closure with the next MutexId. Creates that mutex if the closure returns None,
    /// otherwise returns the value from the closure
    fn mutex_get_or_create<F>(&mut self, existing: F) -> InterpResult<'tcx, MutexId>
    where
        F: FnOnce(&mut MiriInterpCx<'mir, 'tcx>, MutexId) -> InterpResult<'tcx, Option<MutexId>>,
    {
        let this = self.eval_context_mut();
        let next_index = this.machine.threads.sync.mutexes.next_index();
        if let Some(old) = existing(this, next_index)? {
            Ok(old)
        } else {
            let new_index = this.machine.threads.sync.mutexes.push(Default::default());
            assert_eq!(next_index, new_index);
            Ok(new_index)
        }
    }

    #[inline]
    /// Get the id of the thread that currently owns this lock.
    fn mutex_get_owner(&mut self, id: MutexId) -> ThreadId {
        let this = self.eval_context_ref();
        this.machine.threads.sync.mutexes[id].owner.unwrap()
    }

    #[inline]
    /// Check if locked.
    fn mutex_is_locked(&self, id: MutexId) -> bool {
        let this = self.eval_context_ref();
        this.machine.threads.sync.mutexes[id].owner.is_some()
    }

    /// Lock by setting the mutex owner and increasing the lock count.
    fn mutex_lock(&mut self, id: MutexId, thread: ThreadId) {
        let this = self.eval_context_mut();
        let mutex = &mut this.machine.threads.sync.mutexes[id];
        if let Some(current_owner) = mutex.owner {
            assert_eq!(thread, current_owner, "mutex already locked by another thread");
            assert!(
                mutex.lock_count > 0,
                "invariant violation: lock_count == 0 iff the thread is unlocked"
            );
        } else {
            mutex.owner = Some(thread);
        }
        mutex.lock_count = mutex.lock_count.checked_add(1).unwrap();
        if let Some(data_race) = &this.machine.data_race {
            data_race.validate_lock_acquire(&mutex.data_race, thread);
        }
    }

    /// Try unlocking by decreasing the lock count and returning the old lock
    /// count. If the lock count reaches 0, release the lock and potentially
    /// give to a new owner. If the lock was not locked by `expected_owner`,
    /// return `None`.
    fn mutex_unlock(&mut self, id: MutexId, expected_owner: ThreadId) -> Option<usize> {
        let this = self.eval_context_mut();
        let current_span = this.machine.current_span();
        let mutex = &mut this.machine.threads.sync.mutexes[id];
        if let Some(current_owner) = mutex.owner {
            // Mutex is locked.
            if current_owner != expected_owner {
                // Only the owner can unlock the mutex.
                return None;
            }
            let old_lock_count = mutex.lock_count;
            mutex.lock_count = old_lock_count
                .checked_sub(1)
                .expect("invariant violation: lock_count == 0 iff the thread is unlocked");
            if mutex.lock_count == 0 {
                mutex.owner = None;
                // The mutex is completely unlocked. Try transferring ownership
                // to another thread.
                if let Some(data_race) = &this.machine.data_race {
                    data_race.validate_lock_release(
                        &mut mutex.data_race,
                        current_owner,
                        current_span,
                    );
                }
                this.mutex_dequeue_and_lock(id);
            }
            Some(old_lock_count)
        } else {
            // Mutex is not locked.
            None
        }
    }

    /// Put the thread into the queue waiting for the mutex.
    #[inline]
    fn mutex_enqueue_and_block(&mut self, id: MutexId, thread: ThreadId) {
        let this = self.eval_context_mut();
        assert!(this.mutex_is_locked(id), "queing on unlocked mutex");
        this.machine.threads.sync.mutexes[id].queue.push_back(thread);
        this.block_thread(thread);
    }

    /// Provides the closure with the next RwLockId. Creates that RwLock if the closure returns None,
    /// otherwise returns the value from the closure
    #[inline]
    fn rwlock_get_or_create<F>(&mut self, existing: F) -> InterpResult<'tcx, RwLockId>
    where
        F: FnOnce(&mut MiriInterpCx<'mir, 'tcx>, RwLockId) -> InterpResult<'tcx, Option<RwLockId>>,
    {
        let this = self.eval_context_mut();
        let next_index = this.machine.threads.sync.rwlocks.next_index();
        if let Some(old) = existing(this, next_index)? {
            Ok(old)
        } else {
            let new_index = this.machine.threads.sync.rwlocks.push(Default::default());
            assert_eq!(next_index, new_index);
            Ok(new_index)
        }
    }

    #[inline]
    /// Check if locked.
    fn rwlock_is_locked(&self, id: RwLockId) -> bool {
        let this = self.eval_context_ref();
        let rwlock = &this.machine.threads.sync.rwlocks[id];
        trace!(
            "rwlock_is_locked: {:?} writer is {:?} and there are {} reader threads (some of which could hold multiple read locks)",
            id,
            rwlock.writer,
            rwlock.readers.len(),
        );
        rwlock.writer.is_some() || rwlock.readers.is_empty().not()
    }

    /// Check if write locked.
    #[inline]
    fn rwlock_is_write_locked(&self, id: RwLockId) -> bool {
        let this = self.eval_context_ref();
        let rwlock = &this.machine.threads.sync.rwlocks[id];
        trace!("rwlock_is_write_locked: {:?} writer is {:?}", id, rwlock.writer);
        rwlock.writer.is_some()
    }

    /// Read-lock the lock by adding the `reader` the list of threads that own
    /// this lock.
    fn rwlock_reader_lock(&mut self, id: RwLockId, reader: ThreadId) {
        let this = self.eval_context_mut();
        assert!(!this.rwlock_is_write_locked(id), "the lock is write locked");
        trace!("rwlock_reader_lock: {:?} now also held (one more time) by {:?}", id, reader);
        let rwlock = &mut this.machine.threads.sync.rwlocks[id];
        let count = rwlock.readers.entry(reader).or_insert(0);
        *count = count.checked_add(1).expect("the reader counter overflowed");
        if let Some(data_race) = &this.machine.data_race {
            data_race.validate_lock_acquire(&rwlock.data_race, reader);
        }
    }

    /// Try read-unlock the lock for `reader` and potentially give the lock to a new owner.
    /// Returns `true` if succeeded, `false` if this `reader` did not hold the lock.
    fn rwlock_reader_unlock(&mut self, id: RwLockId, reader: ThreadId) -> bool {
        let this = self.eval_context_mut();
        let current_span = this.machine.current_span();
        let rwlock = &mut this.machine.threads.sync.rwlocks[id];
        match rwlock.readers.entry(reader) {
            Entry::Occupied(mut entry) => {
                let count = entry.get_mut();
                assert!(*count > 0, "rwlock locked with count == 0");
                *count -= 1;
                if *count == 0 {
                    trace!("rwlock_reader_unlock: {:?} no longer held by {:?}", id, reader);
                    entry.remove();
                } else {
                    trace!("rwlock_reader_unlock: {:?} held one less time by {:?}", id, reader);
                }
            }
            Entry::Vacant(_) => return false, // we did not even own this lock
        }
        if let Some(data_race) = &this.machine.data_race {
            data_race.validate_lock_release_shared(
                &mut rwlock.data_race_reader,
                reader,
                current_span,
            );
        }

        // The thread was a reader. If the lock is not held any more, give it to a writer.
        if this.rwlock_is_locked(id).not() {
            // All the readers are finished, so set the writer data-race handle to the value
            //  of the union of all reader data race handles, since the set of readers
            //  happen-before the writers
            let rwlock = &mut this.machine.threads.sync.rwlocks[id];
            rwlock.data_race.clone_from(&rwlock.data_race_reader);
            this.rwlock_dequeue_and_lock_writer(id);
        }
        true
    }

    /// Put the reader in the queue waiting for the lock and block it.
    #[inline]
    fn rwlock_enqueue_and_block_reader(&mut self, id: RwLockId, reader: ThreadId) {
        let this = self.eval_context_mut();
        assert!(this.rwlock_is_write_locked(id), "read-queueing on not write locked rwlock");
        this.machine.threads.sync.rwlocks[id].reader_queue.push_back(reader);
        this.block_thread(reader);
    }

    /// Lock by setting the writer that owns the lock.
    #[inline]
    fn rwlock_writer_lock(&mut self, id: RwLockId, writer: ThreadId) {
        let this = self.eval_context_mut();
        assert!(!this.rwlock_is_locked(id), "the rwlock is already locked");
        trace!("rwlock_writer_lock: {:?} now held by {:?}", id, writer);
        let rwlock = &mut this.machine.threads.sync.rwlocks[id];
        rwlock.writer = Some(writer);
        if let Some(data_race) = &this.machine.data_race {
            data_race.validate_lock_acquire(&rwlock.data_race, writer);
        }
    }

    /// Try to unlock by removing the writer.
    #[inline]
    fn rwlock_writer_unlock(&mut self, id: RwLockId, expected_writer: ThreadId) -> bool {
        let this = self.eval_context_mut();
        let current_span = this.machine.current_span();
        let rwlock = &mut this.machine.threads.sync.rwlocks[id];
        if let Some(current_writer) = rwlock.writer {
            if current_writer != expected_writer {
                // Only the owner can unlock the rwlock.
                return false;
            }
            rwlock.writer = None;
            trace!("rwlock_writer_unlock: {:?} unlocked by {:?}", id, expected_writer);
            // Release memory to both reader and writer vector clocks
            //  since this writer happens-before both the union of readers once they are finished
            //  and the next writer
            if let Some(data_race) = &this.machine.data_race {
                data_race.validate_lock_release(
                    &mut rwlock.data_race,
                    current_writer,
                    current_span,
                );
                data_race.validate_lock_release(
                    &mut rwlock.data_race_reader,
                    current_writer,
                    current_span,
                );
            }
            // The thread was a writer.
            //
            // We are prioritizing writers here against the readers. As a
            // result, not only readers can starve writers, but also writers can
            // starve readers.
            if this.rwlock_dequeue_and_lock_writer(id) {
                // Someone got the write lock, nice.
            } else {
                // Give the lock to all readers.
                while this.rwlock_dequeue_and_lock_reader(id) {
                    // Rinse and repeat.
                }
            }
            true
        } else {
            false
        }
    }

    /// Put the writer in the queue waiting for the lock.
    #[inline]
    fn rwlock_enqueue_and_block_writer(&mut self, id: RwLockId, writer: ThreadId) {
        let this = self.eval_context_mut();
        assert!(this.rwlock_is_locked(id), "write-queueing on unlocked rwlock");
        this.machine.threads.sync.rwlocks[id].writer_queue.push_back(writer);
        this.block_thread(writer);
    }

    /// Provides the closure with the next CondvarId. Creates that Condvar if the closure returns None,
    /// otherwise returns the value from the closure
    #[inline]
    fn condvar_get_or_create<F>(&mut self, existing: F) -> InterpResult<'tcx, CondvarId>
    where
        F: FnOnce(
            &mut MiriInterpCx<'mir, 'tcx>,
            CondvarId,
        ) -> InterpResult<'tcx, Option<CondvarId>>,
    {
        let this = self.eval_context_mut();
        let next_index = this.machine.threads.sync.condvars.next_index();
        if let Some(old) = existing(this, next_index)? {
            Ok(old)
        } else {
            let new_index = this.machine.threads.sync.condvars.push(Default::default());
            assert_eq!(next_index, new_index);
            Ok(new_index)
        }
    }

    /// Is the conditional variable awaited?
    #[inline]
    fn condvar_is_awaited(&mut self, id: CondvarId) -> bool {
        let this = self.eval_context_mut();
        !this.machine.threads.sync.condvars[id].waiters.is_empty()
    }

    /// Mark that the thread is waiting on the conditional variable.
    fn condvar_wait(&mut self, id: CondvarId, thread: ThreadId, lock: CondvarLock) {
        let this = self.eval_context_mut();
        let waiters = &mut this.machine.threads.sync.condvars[id].waiters;
        assert!(waiters.iter().all(|waiter| waiter.thread != thread), "thread is already waiting");
        waiters.push_back(CondvarWaiter { thread, lock });
    }

    /// Wake up some thread (if there is any) sleeping on the conditional
    /// variable.
    fn condvar_signal(&mut self, id: CondvarId) -> Option<(ThreadId, CondvarLock)> {
        let this = self.eval_context_mut();
        let current_thread = this.get_active_thread();
        let current_span = this.machine.current_span();
        let condvar = &mut this.machine.threads.sync.condvars[id];
        let data_race = &this.machine.data_race;

        // Each condvar signal happens-before the end of the condvar wake
        if let Some(data_race) = data_race {
            data_race.validate_lock_release(&mut condvar.data_race, current_thread, current_span);
        }
        condvar.waiters.pop_front().map(|waiter| {
            if let Some(data_race) = data_race {
                data_race.validate_lock_acquire(&condvar.data_race, waiter.thread);
            }
            (waiter.thread, waiter.lock)
        })
    }

    #[inline]
    /// Remove the thread from the queue of threads waiting on this conditional variable.
    fn condvar_remove_waiter(&mut self, id: CondvarId, thread: ThreadId) {
        let this = self.eval_context_mut();
        this.machine.threads.sync.condvars[id].waiters.retain(|waiter| waiter.thread != thread);
    }

    fn futex_wait(&mut self, addr: u64, thread: ThreadId, bitset: u32) {
        let this = self.eval_context_mut();
        let futex = &mut this.machine.threads.sync.futexes.entry(addr).or_default();
        let waiters = &mut futex.waiters;
        assert!(waiters.iter().all(|waiter| waiter.thread != thread), "thread is already waiting");
        waiters.push_back(FutexWaiter { thread, bitset });
    }

    fn futex_wake(&mut self, addr: u64, bitset: u32) -> Option<ThreadId> {
        let this = self.eval_context_mut();
        let current_thread = this.get_active_thread();
        let current_span = this.machine.current_span();
        let futex = &mut this.machine.threads.sync.futexes.get_mut(&addr)?;
        let data_race = &this.machine.data_race;

        // Each futex-wake happens-before the end of the futex wait
        if let Some(data_race) = data_race {
            data_race.validate_lock_release(&mut futex.data_race, current_thread, current_span);
        }

        // Wake up the first thread in the queue that matches any of the bits in the bitset.
        futex.waiters.iter().position(|w| w.bitset & bitset != 0).map(|i| {
            let waiter = futex.waiters.remove(i).unwrap();
            if let Some(data_race) = data_race {
                data_race.validate_lock_acquire(&futex.data_race, waiter.thread);
            }
            waiter.thread
        })
    }

    fn futex_remove_waiter(&mut self, addr: u64, thread: ThreadId) {
        let this = self.eval_context_mut();
        if let Some(futex) = this.machine.threads.sync.futexes.get_mut(&addr) {
            futex.waiters.retain(|waiter| waiter.thread != thread);
        }
    }
}
