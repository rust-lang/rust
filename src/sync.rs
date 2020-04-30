use std::collections::{hash_map::Entry, HashMap, VecDeque};
use std::convert::TryFrom;
use std::num::NonZeroU32;
use std::time::Instant;

use rustc_index::vec::{Idx, IndexVec};

use crate::*;

macro_rules! declare_id {
    ($name: ident) => {
        /// 0 is used to indicate that the id was not yet assigned and,
        /// therefore, is not a valid identifier.
        #[derive(Clone, Copy, Debug, PartialOrd, Ord, PartialEq, Eq, Hash)]
        pub struct $name(NonZeroU32);

        impl $name {
            // Panics if `id == 0`.
            pub fn from_u32(id: u32) -> Self {
                Self(NonZeroU32::new(id).unwrap())
            }
        }

        impl Idx for $name {
            fn new(idx: usize) -> Self {
                $name(NonZeroU32::new(u32::try_from(idx).unwrap() + 1).unwrap())
            }
            fn index(self) -> usize {
                usize::try_from(self.0.get() - 1).unwrap()
            }
        }

        impl $name {
            pub fn to_u32_scalar<'tcx>(&self) -> Scalar<Tag> {
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
}

declare_id!(RwLockId);

/// The read-write lock state.
#[derive(Default, Debug)]
struct RwLock {
    /// The writer thread that currently owns the lock.
    writer: Option<ThreadId>,
    /// The readers that currently own the lock and how many times they acquired
    /// the lock.
    readers: HashMap<ThreadId, usize>,
    /// The queue of writer threads waiting for this lock.
    writer_queue: VecDeque<ThreadId>,
    /// The queue of reader threads waiting for this lock.
    reader_queue: VecDeque<ThreadId>,
}

declare_id!(CondvarId);

/// A thread waiting on a conditional variable.
#[derive(Debug)]
struct CondvarWaiter {
    /// The thread that is waiting on this variable.
    thread: ThreadId,
    /// The mutex on which the thread is waiting.
    mutex: MutexId,
    /// The moment in time when the waiter should time out.
    timeout: Option<Instant>,
}

/// The conditional variable state.
#[derive(Default, Debug)]
struct Condvar {
    waiters: VecDeque<CondvarWaiter>,
}

/// The state of all synchronization variables.
#[derive(Default, Debug)]
pub(super) struct SynchronizationState {
    mutexes: IndexVec<MutexId, Mutex>,
    rwlocks: IndexVec<RwLockId, RwLock>,
    condvars: IndexVec<CondvarId, Condvar>,
}

// Public interface to synchronization primitives. Please note that in most
// cases, the function calls are infallible and it is the client's (shim
// implementation's) responsibility to detect and deal with erroneous
// situations.
impl<'mir, 'tcx: 'mir> EvalContextExt<'mir, 'tcx> for crate::MiriEvalContext<'mir, 'tcx> {}
pub trait EvalContextExt<'mir, 'tcx: 'mir>: crate::MiriEvalContextExt<'mir, 'tcx> {
    #[inline]
    /// Create state for a new mutex.
    fn mutex_create(&mut self) -> MutexId {
        let this = self.eval_context_mut();
        this.machine.threads.sync.mutexes.push(Default::default())
    }

    #[inline]
    /// Get the id of the thread that currently owns this lock.
    fn mutex_get_owner(&mut self, id: MutexId) -> ThreadId {
        let this = self.eval_context_ref();
        this.machine.threads.sync.mutexes[id].owner.unwrap()
    }

    #[inline]
    /// Check if locked.
    fn mutex_is_locked(&mut self, id: MutexId) -> bool {
        let this = self.eval_context_mut();
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
    }

    /// Unlock by decreasing the lock count. If the lock count reaches 0, unset
    /// the owner.
    fn mutex_unlock(&mut self, id: MutexId) -> Option<(ThreadId, usize)> {
        let this = self.eval_context_mut();
        let mutex = &mut this.machine.threads.sync.mutexes[id];
        if let Some(current_owner) = mutex.owner {
            mutex.lock_count = mutex
                .lock_count
                .checked_sub(1)
                .expect("invariant violation: lock_count == 0 iff the thread is unlocked");
            if mutex.lock_count == 0 {
                mutex.owner = None;
            }
            Some((current_owner, mutex.lock_count))
        } else {
            None
        }
    }

    #[inline]
    /// Take a thread out the queue waiting for the lock.
    fn mutex_enqueue(&mut self, id: MutexId, thread: ThreadId) {
        let this = self.eval_context_mut();
        this.machine.threads.sync.mutexes[id].queue.push_back(thread);
    }

    #[inline]
    /// Take a thread out the queue waiting for the lock.
    fn mutex_dequeue(&mut self, id: MutexId) -> Option<ThreadId> {
        let this = self.eval_context_mut();
        this.machine.threads.sync.mutexes[id].queue.pop_front()
    }

    #[inline]
    /// Create state for a new read write lock.
    fn rwlock_create(&mut self) -> RwLockId {
        let this = self.eval_context_mut();
        this.machine.threads.sync.rwlocks.push(Default::default())
    }

    #[inline]
    /// Check if locked.
    fn rwlock_is_locked(&mut self, id: RwLockId) -> bool {
        let this = self.eval_context_mut();
        this.machine.threads.sync.rwlocks[id].writer.is_some()
            || !this.machine.threads.sync.rwlocks[id].readers.is_empty()
    }

    #[inline]
    /// Check if write locked.
    fn rwlock_is_write_locked(&mut self, id: RwLockId) -> bool {
        let this = self.eval_context_mut();
        this.machine.threads.sync.rwlocks[id].writer.is_some()
    }

    /// Add a reader that collectively with other readers owns the lock.
    fn rwlock_reader_add(&mut self, id: RwLockId, reader: ThreadId) {
        let this = self.eval_context_mut();
        assert!(!this.rwlock_is_write_locked(id), "the lock is write locked");
        let count = this.machine.threads.sync.rwlocks[id].readers.entry(reader).or_insert(0);
        *count += 1;
    }

    /// Try removing the reader. Returns `true` if succeeded.
    fn rwlock_reader_remove(&mut self, id: RwLockId, reader: ThreadId) -> bool {
        let this = self.eval_context_mut();
        match this.machine.threads.sync.rwlocks[id].readers.entry(reader) {
            Entry::Occupied(mut entry) => {
                let count = entry.get_mut();
                *count -= 1;
                if *count == 0 {
                    entry.remove();
                }
                true
            }
            Entry::Vacant(_) => false,
        }
    }

    #[inline]
    /// Put the reader in the queue waiting for the lock.
    fn rwlock_enqueue_reader(&mut self, id: RwLockId, reader: ThreadId) {
        let this = self.eval_context_mut();
        assert!(this.rwlock_is_write_locked(id), "queueing on not write locked lock");
        this.machine.threads.sync.rwlocks[id].reader_queue.push_back(reader);
    }

    #[inline]
    /// Take the reader out the queue waiting for the lock.
    fn rwlock_dequeue_reader(&mut self, id: RwLockId) -> Option<ThreadId> {
        let this = self.eval_context_mut();
        this.machine.threads.sync.rwlocks[id].reader_queue.pop_front()
    }

    #[inline]
    /// Lock by setting the writer that owns the lock.
    fn rwlock_writer_set(&mut self, id: RwLockId, writer: ThreadId) {
        let this = self.eval_context_mut();
        assert!(!this.rwlock_is_locked(id), "the lock is already locked");
        this.machine.threads.sync.rwlocks[id].writer = Some(writer);
    }

    #[inline]
    /// Try removing the writer.
    fn rwlock_writer_remove(&mut self, id: RwLockId) -> Option<ThreadId> {
        let this = self.eval_context_mut();
        this.machine.threads.sync.rwlocks[id].writer.take()
    }

    #[inline]
    /// Put the writer in the queue waiting for the lock.
    fn rwlock_enqueue_writer(&mut self, id: RwLockId, writer: ThreadId) {
        let this = self.eval_context_mut();
        assert!(this.rwlock_is_locked(id), "queueing on unlocked lock");
        this.machine.threads.sync.rwlocks[id].writer_queue.push_back(writer);
    }

    #[inline]
    /// Take the writer out the queue waiting for the lock.
    fn rwlock_dequeue_writer(&mut self, id: RwLockId) -> Option<ThreadId> {
        let this = self.eval_context_mut();
        this.machine.threads.sync.rwlocks[id].writer_queue.pop_front()
    }

    #[inline]
    /// Create state for a new conditional variable.
    fn condvar_create(&mut self) -> CondvarId {
        let this = self.eval_context_mut();
        this.machine.threads.sync.condvars.push(Default::default())
    }

    #[inline]
    /// Is the conditional variable awaited?
    fn condvar_is_awaited(&mut self, id: CondvarId) -> bool {
        let this = self.eval_context_mut();
        !this.machine.threads.sync.condvars[id].waiters.is_empty()
    }

    /// Mark that the thread is waiting on the conditional variable.
    fn condvar_wait(&mut self, id: CondvarId, thread: ThreadId, mutex: MutexId) {
        let this = self.eval_context_mut();
        let waiters = &mut this.machine.threads.sync.condvars[id].waiters;
        assert!(waiters.iter().all(|waiter| waiter.thread != thread), "thread is already waiting");
        waiters.push_back(CondvarWaiter { thread, mutex, timeout: None });
    }

    /// Wake up some thread (if there is any) sleeping on the conditional
    /// variable.
    fn condvar_signal(&mut self, id: CondvarId) -> Option<(ThreadId, MutexId)> {
        let this = self.eval_context_mut();
        this.machine.threads.sync.condvars[id]
            .waiters
            .pop_front()
            .map(|waiter| (waiter.thread, waiter.mutex))
    }

    #[inline]
    /// Remove the thread from the queue of threads waiting on this conditional variable.
    fn condvar_remove_waiter(&mut self, id: CondvarId, thread: ThreadId) {
        let this = self.eval_context_mut();
        this.machine.threads.sync.condvars[id].waiters.retain(|waiter| waiter.thread != thread);
    }
}
