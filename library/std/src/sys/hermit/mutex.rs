use crate::cell::UnsafeCell;
use crate::collections::VecDeque;
use crate::ffi::c_void;
use crate::hint;
use crate::ops::{Deref, DerefMut, Drop};
use crate::ptr;
use crate::sync::atomic::{AtomicUsize, Ordering};
use crate::sys::hermit::abi;

/// This type provides a lock based on busy waiting to realize mutual exclusion
///
/// # Description
///
/// This structure behaves a lot like a common mutex. There are some differences:
///
/// - By using busy waiting, it can be used outside the runtime.
/// - It is a so called ticket lock and is completely fair.
#[cfg_attr(target_arch = "x86_64", repr(align(128)))]
#[cfg_attr(not(target_arch = "x86_64"), repr(align(64)))]
struct Spinlock<T: ?Sized> {
    queue: AtomicUsize,
    dequeue: AtomicUsize,
    data: UnsafeCell<T>,
}

unsafe impl<T: ?Sized + Send> Sync for Spinlock<T> {}
unsafe impl<T: ?Sized + Send> Send for Spinlock<T> {}

/// A guard to which the protected data can be accessed
///
/// When the guard falls out of scope it will release the lock.
struct SpinlockGuard<'a, T: ?Sized + 'a> {
    dequeue: &'a AtomicUsize,
    data: &'a mut T,
}

impl<T> Spinlock<T> {
    pub const fn new(user_data: T) -> Spinlock<T> {
        Spinlock {
            queue: AtomicUsize::new(0),
            dequeue: AtomicUsize::new(1),
            data: UnsafeCell::new(user_data),
        }
    }

    #[inline]
    fn obtain_lock(&self) {
        let ticket = self.queue.fetch_add(1, Ordering::SeqCst) + 1;
        let mut counter: u16 = 0;
        while self.dequeue.load(Ordering::SeqCst) != ticket {
            counter += 1;
            if counter < 100 {
                hint::spin_loop();
            } else {
                counter = 0;
                unsafe {
                    abi::yield_now();
                }
            }
        }
    }

    #[inline]
    pub unsafe fn lock(&self) -> SpinlockGuard<'_, T> {
        self.obtain_lock();
        SpinlockGuard { dequeue: &self.dequeue, data: &mut *self.data.get() }
    }
}

impl<T: ?Sized + Default> Default for Spinlock<T> {
    fn default() -> Spinlock<T> {
        Spinlock::new(Default::default())
    }
}

impl<'a, T: ?Sized> Deref for SpinlockGuard<'a, T> {
    type Target = T;
    fn deref(&self) -> &T {
        &*self.data
    }
}

impl<'a, T: ?Sized> DerefMut for SpinlockGuard<'a, T> {
    fn deref_mut(&mut self) -> &mut T {
        &mut *self.data
    }
}

impl<'a, T: ?Sized> Drop for SpinlockGuard<'a, T> {
    /// The dropping of the SpinlockGuard will release the lock it was created from.
    fn drop(&mut self) {
        self.dequeue.fetch_add(1, Ordering::SeqCst);
    }
}

/// Realize a priority queue for tasks
struct PriorityQueue {
    queues: [Option<VecDeque<abi::Tid>>; abi::NO_PRIORITIES],
    prio_bitmap: u64,
}

impl PriorityQueue {
    pub const fn new() -> PriorityQueue {
        PriorityQueue {
            queues: [
                None, None, None, None, None, None, None, None, None, None, None, None, None, None,
                None, None, None, None, None, None, None, None, None, None, None, None, None, None,
                None, None, None,
            ],
            prio_bitmap: 0,
        }
    }

    /// Add a task id by its priority to the queue
    pub fn push(&mut self, prio: abi::Priority, id: abi::Tid) {
        let i: usize = prio.into().into();
        self.prio_bitmap |= (1 << i) as u64;
        if let Some(queue) = &mut self.queues[i] {
            queue.push_back(id);
        } else {
            let mut queue = VecDeque::new();
            queue.push_back(id);
            self.queues[i] = Some(queue);
        }
    }

    fn pop_from_queue(&mut self, queue_index: usize) -> Option<abi::Tid> {
        if let Some(queue) = &mut self.queues[queue_index] {
            let id = queue.pop_front();

            if queue.is_empty() {
                self.prio_bitmap &= !(1 << queue_index as u64);
            }

            id
        } else {
            None
        }
    }

    /// Pop the task handle with the highest priority from the queue
    pub fn pop(&mut self) -> Option<abi::Tid> {
        for i in 0..abi::NO_PRIORITIES {
            if self.prio_bitmap & (1 << i) != 0 {
                return self.pop_from_queue(i);
            }
        }

        None
    }
}

struct MutexInner {
    locked: bool,
    blocked_task: PriorityQueue,
}

impl MutexInner {
    pub const fn new() -> MutexInner {
        MutexInner { locked: false, blocked_task: PriorityQueue::new() }
    }
}

pub struct Mutex {
    inner: Spinlock<MutexInner>,
}

pub type MovableMutex = Mutex;

unsafe impl Send for Mutex {}
unsafe impl Sync for Mutex {}

impl Mutex {
    pub const fn new() -> Mutex {
        Mutex { inner: Spinlock::new(MutexInner::new()) }
    }

    #[inline]
    pub unsafe fn init(&mut self) {
        self.inner = Spinlock::new(MutexInner::new());
    }

    #[inline]
    pub unsafe fn lock(&self) {
        loop {
            let mut guard = self.inner.lock();
            if guard.locked == false {
                guard.locked = true;
                return;
            } else {
                let prio = abi::get_priority();
                let id = abi::getpid();

                guard.blocked_task.push(prio, id);
                abi::block_current_task();
                drop(guard);
                abi::yield_now();
            }
        }
    }

    #[inline]
    pub unsafe fn unlock(&self) {
        let mut guard = self.inner.lock();
        guard.locked = false;
        if let Some(tid) = guard.blocked_task.pop() {
            abi::wakeup_task(tid);
        }
    }

    #[inline]
    pub unsafe fn try_lock(&self) -> bool {
        let mut guard = self.inner.lock();
        if guard.locked == false {
            guard.locked = true;
        }
        guard.locked
    }

    #[inline]
    pub unsafe fn destroy(&self) {}
}

pub struct ReentrantMutex {
    inner: *const c_void,
}

impl ReentrantMutex {
    pub const unsafe fn uninitialized() -> ReentrantMutex {
        ReentrantMutex { inner: ptr::null() }
    }

    #[inline]
    pub unsafe fn init(&self) {
        let _ = abi::recmutex_init(&self.inner as *const *const c_void as *mut _);
    }

    #[inline]
    pub unsafe fn lock(&self) {
        let _ = abi::recmutex_lock(self.inner);
    }

    #[inline]
    pub unsafe fn try_lock(&self) -> bool {
        true
    }

    #[inline]
    pub unsafe fn unlock(&self) {
        let _ = abi::recmutex_unlock(self.inner);
    }

    #[inline]
    pub unsafe fn destroy(&self) {
        let _ = abi::recmutex_destroy(self.inner);
    }
}
