/// A simple queue implementation for synchronization primitives.
///
/// This queue is used to implement condition variable and mutexes.
///
/// Users of this API are expected to use the `WaitVariable<T>` type. Since
/// that type is not `Sync`, it needs to be protected by e.g., a `SpinMutex` to
/// allow shared access.
///
/// Since userspace may send spurious wake-ups, the wakeup event state is
/// recorded in the enclave. The wakeup event state is protected by a spinlock.
/// The queue and associated wait state are stored in a `WaitVariable`.

use crate::ops::{Deref, DerefMut};
use crate::num::NonZeroUsize;

use fortanix_sgx_abi::{Tcs, EV_UNPARK, WAIT_INDEFINITE};
use super::abi::usercalls;
use super::abi::thread;

use self::unsafe_list::{UnsafeList, UnsafeListEntry};
pub use self::spin_mutex::{SpinMutex, SpinMutexGuard, try_lock_or_false};

/// An queue entry in a `WaitQueue`.
struct WaitEntry {
    /// TCS address of the thread that is waiting
    tcs: Tcs,
    /// Whether this thread has been notified to be awoken
    wake: bool
}

/// Data stored with a `WaitQueue` alongside it. This ensures accesses to the
/// queue and the data are synchronized, since the type itself is not `Sync`.
///
/// Consumers of this API should use a synchronization primitive for shared
/// access, such as `SpinMutex`.
#[derive(Default)]
pub struct WaitVariable<T> {
    queue: WaitQueue,
    lock: T
}

impl<T> WaitVariable<T> {
    pub const fn new(var: T) -> Self {
        WaitVariable {
            queue: WaitQueue::new(),
            lock: var
        }
    }

    pub fn queue_empty(&self) -> bool {
        self.queue.is_empty()
    }

    pub fn lock_var(&self) -> &T {
        &self.lock
    }

    pub fn lock_var_mut(&mut self) -> &mut T {
        &mut self.lock
    }
}

#[derive(Copy, Clone)]
pub enum NotifiedTcs {
    Single(Tcs),
    All { count: NonZeroUsize }
}

/// An RAII guard that will notify a set of target threads as well as unlock
/// a mutex on drop.
pub struct WaitGuard<'a, T: 'a> {
    mutex_guard: Option<SpinMutexGuard<'a, WaitVariable<T>>>,
    notified_tcs: NotifiedTcs
}

/// A queue of threads that are waiting on some synchronization primitive.
///
/// `UnsafeList` entries are allocated on the waiting thread's stack. This
/// avoids any global locking that might happen in the heap allocator. This is
/// safe because the waiting thread will not return from that stack frame until
/// after it is notified. The notifying thread ensures to clean up any
/// references to the list entries before sending the wakeup event.
pub struct WaitQueue {
    // We use an inner Mutex here to protect the data in the face of spurious
    // wakeups.
    inner: UnsafeList<SpinMutex<WaitEntry>>,
}
unsafe impl Send for WaitQueue {}

impl Default for WaitQueue {
    fn default() -> Self {
        Self::new()
    }
}

impl<'a, T> WaitGuard<'a, T> {
    /// Returns which TCSes will be notified when this guard drops.
    pub fn notified_tcs(&self) -> NotifiedTcs {
        self.notified_tcs
    }
}

impl<'a, T> Deref for WaitGuard<'a, T> {
    type Target = SpinMutexGuard<'a, WaitVariable<T>>;

    fn deref(&self) -> &Self::Target {
        self.mutex_guard.as_ref().unwrap()
    }
}

impl<'a, T> DerefMut for WaitGuard<'a, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.mutex_guard.as_mut().unwrap()
    }
}

impl<'a, T> Drop for WaitGuard<'a, T> {
    fn drop(&mut self) {
        drop(self.mutex_guard.take());
        let target_tcs = match self.notified_tcs {
            NotifiedTcs::Single(tcs) => Some(tcs),
            NotifiedTcs::All { .. } => None
        };
        rtunwrap!(Ok, usercalls::send(EV_UNPARK, target_tcs));
    }
}

impl WaitQueue {
    pub const fn new() -> Self {
        WaitQueue {
            inner: UnsafeList::new()
        }
    }

    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Adds the calling thread to the `WaitVariable`'s wait queue, then wait
    /// until a wakeup event.
    ///
    /// This function does not return until this thread has been awoken.
    pub fn wait<T>(mut guard: SpinMutexGuard<'_, WaitVariable<T>>) {
        // very unsafe: check requirements of UnsafeList::push
        unsafe {
            let mut entry = UnsafeListEntry::new(SpinMutex::new(WaitEntry {
                tcs: thread::current(),
                wake: false
            }));
            let entry = guard.queue.inner.push(&mut entry);
            drop(guard);
            while !entry.lock().wake {
                // don't panic, this would invalidate `entry` during unwinding
                let eventset = rtunwrap!(Ok, usercalls::wait(EV_UNPARK, WAIT_INDEFINITE));
                rtassert!(eventset & EV_UNPARK == EV_UNPARK);
            }
        }
    }

    /// Either find the next waiter on the wait queue, or return the mutex
    /// guard unchanged.
    ///
    /// If a waiter is found, a `WaitGuard` is returned which will notify the
    /// waiter when it is dropped.
    pub fn notify_one<T>(mut guard: SpinMutexGuard<'_, WaitVariable<T>>)
        -> Result<WaitGuard<'_, T>, SpinMutexGuard<'_, WaitVariable<T>>>
    {
        unsafe {
            if let Some(entry) = guard.queue.inner.pop() {
                let mut entry_guard = entry.lock();
                let tcs = entry_guard.tcs;
                entry_guard.wake = true;
                drop(entry);
                Ok(WaitGuard {
                    mutex_guard: Some(guard),
                    notified_tcs: NotifiedTcs::Single(tcs)
                })
            } else {
                Err(guard)
            }
        }
    }

    /// Either find any and all waiters on the wait queue, or return the mutex
    /// guard unchanged.
    ///
    /// If at least one waiter is found, a `WaitGuard` is returned which will
    /// notify all waiters when it is dropped.
    pub fn notify_all<T>(mut guard: SpinMutexGuard<'_, WaitVariable<T>>)
        -> Result<WaitGuard<'_, T>, SpinMutexGuard<'_, WaitVariable<T>>>
    {
        unsafe {
            let mut count = 0;
            while let Some(entry) = guard.queue.inner.pop() {
                count += 1;
                let mut entry_guard = entry.lock();
                entry_guard.wake = true;
            }
            if let Some(count) = NonZeroUsize::new(count) {
                Ok(WaitGuard {
                    mutex_guard: Some(guard),
                    notified_tcs: NotifiedTcs::All { count }
                })
            } else {
                Err(guard)
            }
        }
    }
}

/// A doubly-linked list where callers are in charge of memory allocation
/// of the nodes in the list.
mod unsafe_list {
    use crate::ptr::NonNull;
    use crate::mem;

    pub struct UnsafeListEntry<T> {
        next: NonNull<UnsafeListEntry<T>>,
        prev: NonNull<UnsafeListEntry<T>>,
        value: Option<T>
    }

    impl<T> UnsafeListEntry<T> {
        fn dummy() -> Self {
            UnsafeListEntry {
                next: NonNull::dangling(),
                prev: NonNull::dangling(),
                value: None
            }
        }

        pub fn new(value: T) -> Self {
            UnsafeListEntry {
                value: Some(value),
                ..Self::dummy()
            }
        }
    }

    pub struct UnsafeList<T> {
        head_tail: NonNull<UnsafeListEntry<T>>,
        head_tail_entry: Option<UnsafeListEntry<T>>,
    }

    impl<T> UnsafeList<T> {
        pub const fn new() -> Self {
            unsafe {
                UnsafeList {
                    head_tail: NonNull::new_unchecked(1 as _),
                    head_tail_entry: None
                }
            }
        }

        unsafe fn init(&mut self) {
            if self.head_tail_entry.is_none() {
                self.head_tail_entry = Some(UnsafeListEntry::dummy());
                self.head_tail = NonNull::new_unchecked(self.head_tail_entry.as_mut().unwrap());
                self.head_tail.as_mut().next = self.head_tail;
                self.head_tail.as_mut().prev = self.head_tail;
            }
        }

        pub fn is_empty(&self) -> bool {
            unsafe {
                if self.head_tail_entry.is_some() {
                    let first = self.head_tail.as_ref().next;
                    if first == self.head_tail {
                        // ,-------> /---------\ next ---,
                        // |         |head_tail|         |
                        // `--- prev \---------/ <-------`
                        rtassert!(self.head_tail.as_ref().prev == first);
                        true
                    } else {
                        false
                    }
                } else {
                    true
                }
            }
        }

        /// Pushes an entry onto the back of the list.
        ///
        /// # Safety
        ///
        /// The entry must remain allocated until the entry is removed from the
        /// list AND the caller who popped is done using the entry. Special
        /// care must be taken in the caller of `push` to ensure unwinding does
        /// not destroy the stack frame containing the entry.
        pub unsafe fn push<'a>(&mut self, entry: &'a mut UnsafeListEntry<T>) -> &'a T {
            self.init();

            // BEFORE:
            //     /---------\ next ---> /---------\
            // ... |prev_tail|           |head_tail| ...
            //     \---------/ <--- prev \---------/
            //
            // AFTER:
            //     /---------\ next ---> /-----\ next ---> /---------\
            // ... |prev_tail|           |entry|           |head_tail| ...
            //     \---------/ <--- prev \-----/ <--- prev \---------/
            let mut entry = NonNull::new_unchecked(entry);
            let mut prev_tail = mem::replace(&mut self.head_tail.as_mut().prev, entry);
            entry.as_mut().prev = prev_tail;
            entry.as_mut().next = self.head_tail;
            prev_tail.as_mut().next = entry;
            // unwrap ok: always `Some` on non-dummy entries
            (*entry.as_ptr()).value.as_ref().unwrap()
        }

        /// Pops an entry from the front of the list.
        ///
        /// # Safety
        ///
        /// The caller must make sure to synchronize ending the borrow of the
        /// return value and deallocation of the containing entry.
        pub unsafe fn pop<'a>(&mut self) -> Option<&'a T> {
            self.init();

            if self.is_empty() {
                None
            } else {
                // BEFORE:
                //     /---------\ next ---> /-----\ next ---> /------\
                // ... |head_tail|           |first|           |second| ...
                //     \---------/ <--- prev \-----/ <--- prev \------/
                //
                // AFTER:
                //     /---------\ next ---> /------\
                // ... |head_tail|           |second| ...
                //     \---------/ <--- prev \------/
                let mut first = self.head_tail.as_mut().next;
                let mut second = first.as_mut().next;
                self.head_tail.as_mut().next = second;
                second.as_mut().prev = self.head_tail;
                first.as_mut().next = NonNull::dangling();
                first.as_mut().prev = NonNull::dangling();
                // unwrap ok: always `Some` on non-dummy entries
                Some((*first.as_ptr()).value.as_ref().unwrap())
            }
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use crate::cell::Cell;

        unsafe fn assert_empty<T>(list: &mut UnsafeList<T>) {
            assert!(list.pop().is_none(), "assertion failed: list is not empty");
        }

        #[test]
        fn init_empty() {
            unsafe {
                assert_empty(&mut UnsafeList::<i32>::new());
            }
        }

        #[test]
        fn push_pop() {
            unsafe {
                let mut node = UnsafeListEntry::new(1234);
                let mut list = UnsafeList::new();
                assert_eq!(list.push(&mut node), &1234);
                assert_eq!(list.pop().unwrap(), &1234);
                assert_empty(&mut list);
            }
        }

        #[test]
        fn complex_pushes_pops() {
            unsafe {
                let mut node1 = UnsafeListEntry::new(1234);
                let mut node2 = UnsafeListEntry::new(4567);
                let mut node3 = UnsafeListEntry::new(9999);
                let mut node4 = UnsafeListEntry::new(8642);
                let mut list = UnsafeList::new();
                list.push(&mut node1);
                list.push(&mut node2);
                assert_eq!(list.pop().unwrap(), &1234);
                list.push(&mut node3);
                assert_eq!(list.pop().unwrap(), &4567);
                assert_eq!(list.pop().unwrap(), &9999);
                assert_empty(&mut list);
                list.push(&mut node4);
                assert_eq!(list.pop().unwrap(), &8642);
                assert_empty(&mut list);
            }
        }

        #[test]
        fn cell() {
            unsafe {
                let mut node = UnsafeListEntry::new(Cell::new(0));
                let mut list = UnsafeList::new();
                let noderef = list.push(&mut node);
                assert_eq!(noderef.get(), 0);
                list.pop().unwrap().set(1);
                assert_empty(&mut list);
                assert_eq!(noderef.get(), 1);
            }
        }
    }
}

/// Trivial spinlock-based implementation of `sync::Mutex`.
// FIXME: Perhaps use Intel TSX to avoid locking?
mod spin_mutex {
    use crate::cell::UnsafeCell;
    use crate::sync::atomic::{AtomicBool, Ordering, spin_loop_hint};
    use crate::ops::{Deref, DerefMut};

    #[derive(Default)]
    pub struct SpinMutex<T> {
        value: UnsafeCell<T>,
        lock: AtomicBool,
    }

    unsafe impl<T: Send> Send for SpinMutex<T> {}
    unsafe impl<T: Send> Sync for SpinMutex<T> {}

    pub struct SpinMutexGuard<'a, T: 'a> {
        mutex: &'a SpinMutex<T>,
    }

    impl<'a, T> !Send for SpinMutexGuard<'a, T> {}
    unsafe impl<'a, T: Sync> Sync for SpinMutexGuard<'a, T> {}

    impl<T> SpinMutex<T> {
        pub const fn new(value: T) -> Self {
            SpinMutex {
                value: UnsafeCell::new(value),
                lock: AtomicBool::new(false)
            }
        }

        #[inline(always)]
        pub fn lock(&self) -> SpinMutexGuard<'_, T> {
            loop {
                match self.try_lock() {
                    None => while self.lock.load(Ordering::Relaxed) {
                        spin_loop_hint()
                    },
                    Some(guard) => return guard
                }
            }
        }

        #[inline(always)]
        pub fn try_lock(&self) -> Option<SpinMutexGuard<'_, T>> {
            if !self.lock.compare_and_swap(false, true, Ordering::Acquire) {
                Some(SpinMutexGuard {
                    mutex: self,
                })
            } else {
                None
            }
        }
    }

    /// Lock the Mutex or return false.
    pub macro try_lock_or_false {
        ($e:expr) => {
            if let Some(v) = $e.try_lock() {
                v
            } else {
                return false
            }
        }
    }

    impl<'a, T> Deref for SpinMutexGuard<'a, T> {
        type Target = T;

        fn deref(&self) -> &T {
            unsafe {
                &*self.mutex.value.get()
            }
        }
    }

    impl<'a, T> DerefMut for SpinMutexGuard<'a, T> {
        fn deref_mut(&mut self) -> &mut T {
            unsafe {
                &mut*self.mutex.value.get()
            }
        }
    }

    impl<'a, T> Drop for SpinMutexGuard<'a, T> {
        fn drop(&mut self) {
            self.mutex.lock.store(false, Ordering::Release)
        }
    }

    #[cfg(test)]
    mod tests {
        #![allow(deprecated)]

        use super::*;
        use crate::sync::Arc;
        use crate::thread;
        use crate::time::{SystemTime, Duration};

        #[test]
        fn sleep() {
            let mutex = Arc::new(SpinMutex::<i32>::default());
            let mutex2 = mutex.clone();
            let guard = mutex.lock();
            let t1 = thread::spawn(move || {
                *mutex2.lock() = 1;
            });

            // "sleep" for 50ms
            // FIXME: https://github.com/fortanix/rust-sgx/issues/31
            let start = SystemTime::now();
            let max = Duration::from_millis(50);
            while start.elapsed().unwrap() < max {}

            assert_eq!(*guard, 0);
            drop(guard);
            t1.join().unwrap();
            assert_eq!(*mutex.lock(), 1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sync::Arc;
    use crate::thread;

    #[test]
    fn queue() {
        let wq = Arc::new(SpinMutex::<WaitVariable<()>>::default());
        let wq2 = wq.clone();

        let locked = wq.lock();

        let t1 = thread::spawn(move || {
            // if we obtain the lock, the main thread should be waiting
            assert!(WaitQueue::notify_one(wq2.lock()).is_ok());
        });

        WaitQueue::wait(locked);

        t1.join().unwrap();
    }
}
