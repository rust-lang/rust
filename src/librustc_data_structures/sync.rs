//! This module defines types which are thread safe if cfg!(parallel_compiler) is true.
//!
//! `Lrc` is an alias of either Rc or Arc.
//!
//! `Lock` is a mutex.
//! It internally uses `parking_lot::Mutex` if cfg!(parallel_compiler) is true,
//! `RefCell` otherwise.
//!
//! `RwLock` is a read-write lock.
//! It internally uses `parking_lot::RwLock` if cfg!(parallel_compiler) is true,
//! `RefCell` otherwise.
//!
//! `MTLock` is a mutex which disappears if cfg!(parallel_compiler) is false.
//!
//! `MTRef` is a immutable reference if cfg!(parallel_compiler), and an mutable reference otherwise.
//!
//! `rustc_erase_owner!` erases a OwningRef owner into Erased or Erased + Send + Sync
//! depending on the value of cfg!(parallel_compiler).

use std::collections::HashMap;
use std::hash::{Hash, BuildHasher};
use std::marker::PhantomData;
use std::ops::{Deref, DerefMut};
use crate::owning_ref::{Erased, OwningRef};

pub fn serial_join<A, B, RA, RB>(oper_a: A, oper_b: B) -> (RA, RB)
    where A: FnOnce() -> RA,
          B: FnOnce() -> RB
{
    (oper_a(), oper_b())
}

pub struct SerialScope;

impl SerialScope {
    pub fn spawn<F>(&self, f: F)
        where F: FnOnce(&SerialScope)
    {
        f(self)
    }
}

pub fn serial_scope<F, R>(f: F) -> R
    where F: FnOnce(&SerialScope) -> R
{
    f(&SerialScope)
}

pub use std::sync::atomic::Ordering::SeqCst;
pub use std::sync::atomic::Ordering;

cfg_if! {
    if #[cfg(not(parallel_compiler))] {
        pub auto trait Send {}
        pub auto trait Sync {}

        impl<T: ?Sized> Send for T {}
        impl<T: ?Sized> Sync for T {}

        #[macro_export]
        macro_rules! rustc_erase_owner {
            ($v:expr) => {
                $v.erase_owner()
            }
        }

        use std::ops::Add;
        use std::panic::{resume_unwind, catch_unwind, AssertUnwindSafe};

        #[derive(Debug)]
        pub struct Atomic<T: Copy>(Cell<T>);

        impl<T: Copy> Atomic<T> {
            #[inline]
            pub fn new(v: T) -> Self {
                Atomic(Cell::new(v))
            }
        }

        impl<T: Copy + PartialEq> Atomic<T> {
            pub fn into_inner(self) -> T {
                self.0.into_inner()
            }

            #[inline]
            pub fn load(&self, _: Ordering) -> T {
                self.0.get()
            }

            #[inline]
            pub fn store(&self, val: T, _: Ordering) {
                self.0.set(val)
            }

            pub fn swap(&self, val: T, _: Ordering) -> T {
                self.0.replace(val)
            }

            pub fn compare_exchange(&self,
                                    current: T,
                                    new: T,
                                    _: Ordering,
                                    _: Ordering)
                                    -> Result<T, T> {
                let read = self.0.get();
                if read == current {
                    self.0.set(new);
                    Ok(read)
                } else {
                    Err(read)
                }
            }
        }

        impl<T: Add<Output=T> + Copy> Atomic<T> {
            pub fn fetch_add(&self, val: T, _: Ordering) -> T {
                let old = self.0.get();
                self.0.set(old + val);
                old
            }
        }

        pub type AtomicUsize = Atomic<usize>;
        pub type AtomicBool = Atomic<bool>;
        pub type AtomicU32 = Atomic<u32>;
        pub type AtomicU64 = Atomic<u64>;

        pub use self::serial_join as join;
        pub use self::serial_scope as scope;

        #[macro_export]
        macro_rules! parallel {
            ($($blocks:tt),*) => {
                // We catch panics here ensuring that all the blocks execute.
                // This makes behavior consistent with the parallel compiler.
                let mut panic = None;
                $(
                    if let Err(p) = ::std::panic::catch_unwind(
                        ::std::panic::AssertUnwindSafe(|| $blocks)
                    ) {
                        if panic.is_none() {
                            panic = Some(p);
                        }
                    }
                )*
                if let Some(panic) = panic {
                    ::std::panic::resume_unwind(panic);
                }
            }
        }

        pub use std::iter::Iterator as ParallelIterator;

        pub fn par_iter<T: IntoIterator>(t: T) -> T::IntoIter {
            t.into_iter()
        }

        pub fn par_for_each_in<T: IntoIterator>(
            t: T,
            for_each:
                impl Fn(<<T as IntoIterator>::IntoIter as Iterator>::Item) + Sync + Send
        ) {
            // We catch panics here ensuring that all the loop iterations execute.
            // This makes behavior consistent with the parallel compiler.
            let mut panic = None;
            t.into_iter().for_each(|i| {
                if let Err(p) = catch_unwind(AssertUnwindSafe(|| for_each(i))) {
                    if panic.is_none() {
                        panic = Some(p);
                    }
                }
            });
            if let Some(panic) = panic {
                resume_unwind(panic);
            }
        }

        pub type MetadataRef = OwningRef<Box<dyn Erased>, [u8]>;

        pub use std::rc::Rc as Lrc;
        pub use std::rc::Weak as Weak;
        pub use std::cell::Ref as ReadGuard;
        pub use std::cell::Ref as MappedReadGuard;
        pub use std::cell::RefMut as WriteGuard;
        pub use std::cell::RefMut as MappedWriteGuard;
        pub use std::cell::RefMut as LockGuard;
        pub use std::cell::RefMut as MappedLockGuard;

        use std::cell::RefCell as InnerRwLock;
        use std::cell::RefCell as InnerLock;

        use std::cell::Cell;

        #[derive(Debug)]
        pub struct WorkerLocal<T>(OneThread<T>);

        impl<T> WorkerLocal<T> {
            /// Creates a new worker local where the `initial` closure computes the
            /// value this worker local should take for each thread in the thread pool.
            #[inline]
            pub fn new<F: FnMut(usize) -> T>(mut f: F) -> WorkerLocal<T> {
                WorkerLocal(OneThread::new(f(0)))
            }

            /// Returns the worker-local value for each thread
            #[inline]
            pub fn into_inner(self) -> Vec<T> {
                vec![OneThread::into_inner(self.0)]
            }
        }

        impl<T> Deref for WorkerLocal<T> {
            type Target = T;

            #[inline(always)]
            fn deref(&self) -> &T {
                &*self.0
            }
        }

        pub type MTRef<'a, T> = &'a mut T;

        #[derive(Debug, Default)]
        pub struct MTLock<T>(T);

        impl<T> MTLock<T> {
            #[inline(always)]
            pub fn new(inner: T) -> Self {
                MTLock(inner)
            }

            #[inline(always)]
            pub fn into_inner(self) -> T {
                self.0
            }

            #[inline(always)]
            pub fn get_mut(&mut self) -> &mut T {
                &mut self.0
            }

            #[inline(always)]
            pub fn lock(&self) -> &T {
                &self.0
            }

            #[inline(always)]
            pub fn lock_mut(&mut self) -> &mut T {
                &mut self.0
            }
        }

        // FIXME: Probably a bad idea (in the threaded case)
        impl<T: Clone> Clone for MTLock<T> {
            #[inline]
            fn clone(&self) -> Self {
                MTLock(self.0.clone())
            }
        }
    } else {
        pub use std::marker::Send as Send;
        pub use std::marker::Sync as Sync;

        pub use parking_lot::RwLockReadGuard as ReadGuard;
        pub use parking_lot::MappedRwLockReadGuard as MappedReadGuard;
        pub use parking_lot::RwLockWriteGuard as WriteGuard;
        pub use parking_lot::MappedRwLockWriteGuard as MappedWriteGuard;

        pub use parking_lot::MutexGuard as LockGuard;
        pub use parking_lot::MappedMutexGuard as MappedLockGuard;

        pub use std::sync::atomic::{AtomicBool, AtomicUsize, AtomicU32, AtomicU64};

        pub use std::sync::Arc as Lrc;
        pub use std::sync::Weak as Weak;

        pub type MTRef<'a, T> = &'a T;

        #[derive(Debug, Default)]
        pub struct MTLock<T>(Lock<T>);

        impl<T> MTLock<T> {
            #[inline(always)]
            pub fn new(inner: T) -> Self {
                MTLock(Lock::new(inner))
            }

            #[inline(always)]
            pub fn into_inner(self) -> T {
                self.0.into_inner()
            }

            #[inline(always)]
            pub fn get_mut(&mut self) -> &mut T {
                self.0.get_mut()
            }

            #[inline(always)]
            pub fn lock(&self) -> LockGuard<'_, T> {
                self.0.lock()
            }

            #[inline(always)]
            pub fn lock_mut(&self) -> LockGuard<'_, T> {
                self.lock()
            }
        }

        use parking_lot::Mutex as InnerLock;
        use parking_lot::RwLock as InnerRwLock;

        use std;
        use std::thread;
        pub use rayon::{join, scope};

        /// Runs a list of blocks in parallel. The first block is executed immediately on
        /// the current thread. Use that for the longest running block.
        #[macro_export]
        macro_rules! parallel {
            (impl $fblock:tt [$($c:tt,)*] [$block:tt $(, $rest:tt)*]) => {
                parallel!(impl $fblock [$block, $($c,)*] [$($rest),*])
            };
            (impl $fblock:tt [$($blocks:tt,)*] []) => {
                ::rustc_data_structures::sync::scope(|s| {
                    $(
                        s.spawn(|_| $blocks);
                    )*
                    $fblock;
                })
            };
            ($fblock:tt, $($blocks:tt),*) => {
                // Reverse the order of the later blocks since Rayon executes them in reverse order
                // when using a single thread. This ensures the execution order matches that
                // of a single threaded rustc
                parallel!(impl $fblock [] [$($blocks),*]);
            };
        }

        pub use rayon_core::WorkerLocal;

        pub use rayon::iter::ParallelIterator;
        use rayon::iter::IntoParallelIterator;

        pub fn par_iter<T: IntoParallelIterator>(t: T) -> T::Iter {
            t.into_par_iter()
        }

        pub fn par_for_each_in<T: IntoParallelIterator>(
            t: T,
            for_each: impl Fn(
                <<T as IntoParallelIterator>::Iter as ParallelIterator>::Item
            ) + Sync + Send
        ) {
            t.into_par_iter().for_each(for_each)
        }

        pub type MetadataRef = OwningRef<Box<dyn Erased + Send + Sync>, [u8]>;

        /// This makes locks panic if they are already held.
        /// It is only useful when you are running in a single thread
        const ERROR_CHECKING: bool = false;

        #[macro_export]
        macro_rules! rustc_erase_owner {
            ($v:expr) => {{
                let v = $v;
                ::rustc_data_structures::sync::assert_send_val(&v);
                v.erase_send_sync_owner()
            }}
        }
    }
}

pub fn assert_sync<T: ?Sized + Sync>() {}
pub fn assert_send<T: ?Sized + Send>() {}
pub fn assert_send_val<T: ?Sized + Send>(_t: &T) {}
pub fn assert_send_sync_val<T: ?Sized + Sync + Send>(_t: &T) {}

pub trait HashMapExt<K, V> {
    /// Same as HashMap::insert, but it may panic if there's already an
    /// entry for `key` with a value not equal to `value`
    fn insert_same(&mut self, key: K, value: V);
}

impl<K: Eq + Hash, V: Eq, S: BuildHasher> HashMapExt<K, V> for HashMap<K, V, S> {
    fn insert_same(&mut self, key: K, value: V) {
        self.entry(key).and_modify(|old| assert!(*old == value)).or_insert(value);
    }
}

/// A type whose inner value can be written once and then will stay read-only
// This contains a PhantomData<T> since this type conceptually owns a T outside the Mutex once
// initialized. This ensures that Once<T> is Sync only if T is. If we did not have PhantomData<T>
// we could send a &Once<Cell<bool>> to multiple threads and call `get` on it to get access
// to &Cell<bool> on those threads.
pub struct Once<T>(Lock<Option<T>>, PhantomData<T>);

impl<T> Once<T> {
    /// Creates an Once value which is uninitialized
    #[inline(always)]
    pub fn new() -> Self {
        Once(Lock::new(None), PhantomData)
    }

    /// Consumes the value and returns Some(T) if it was initialized
    #[inline(always)]
    pub fn into_inner(self) -> Option<T> {
        self.0.into_inner()
    }

    /// Tries to initialize the inner value to `value`.
    /// Returns `None` if the inner value was uninitialized and `value` was consumed setting it
    /// otherwise if the inner value was already set it returns `value` back to the caller
    #[inline]
    pub fn try_set(&self, value: T) -> Option<T> {
        let mut lock = self.0.lock();
        if lock.is_some() {
            return Some(value);
        }
        *lock = Some(value);
        None
    }

    /// Tries to initialize the inner value to `value`.
    /// Returns `None` if the inner value was uninitialized and `value` was consumed setting it
    /// otherwise if the inner value was already set it asserts that `value` is equal to the inner
    /// value and then returns `value` back to the caller
    #[inline]
    pub fn try_set_same(&self, value: T) -> Option<T> where T: Eq {
        let mut lock = self.0.lock();
        if let Some(ref inner) = *lock {
            assert!(*inner == value);
            return Some(value);
        }
        *lock = Some(value);
        None
    }

    /// Tries to initialize the inner value to `value` and panics if it was already initialized
    #[inline]
    pub fn set(&self, value: T) {
        assert!(self.try_set(value).is_none());
    }

    /// Tries to initialize the inner value by calling the closure while ensuring that no-one else
    /// can access the value in the mean time by holding a lock for the duration of the closure.
    /// If the value was already initialized the closure is not called and `false` is returned,
    /// otherwise if the value from the closure initializes the inner value, `true` is returned
    #[inline]
    pub fn init_locking<F: FnOnce() -> T>(&self, f: F) -> bool {
        let mut lock = self.0.lock();
        if lock.is_some() {
            return false;
        }
        *lock = Some(f());
        true
    }

    /// Tries to initialize the inner value by calling the closure without ensuring that no-one
    /// else can access it. This mean when this is called from multiple threads, multiple
    /// closures may concurrently be computing a value which the inner value should take.
    /// Only one of these closures are used to actually initialize the value.
    /// If some other closure already set the value,
    /// we return the value our closure computed wrapped in a `Option`.
    /// If our closure set the value, `None` is returned.
    /// If the value is already initialized, the closure is not called and `None` is returned.
    #[inline]
    pub fn init_nonlocking<F: FnOnce() -> T>(&self, f: F) -> Option<T> {
        if self.0.lock().is_some() {
            None
        } else {
            self.try_set(f())
        }
    }

    /// Tries to initialize the inner value by calling the closure without ensuring that no-one
    /// else can access it. This mean when this is called from multiple threads, multiple
    /// closures may concurrently be computing a value which the inner value should take.
    /// Only one of these closures are used to actually initialize the value.
    /// If some other closure already set the value, we assert that it our closure computed
    /// a value equal to the value already set and then
    /// we return the value our closure computed wrapped in a `Option`.
    /// If our closure set the value, `None` is returned.
    /// If the value is already initialized, the closure is not called and `None` is returned.
    #[inline]
    pub fn init_nonlocking_same<F: FnOnce() -> T>(&self, f: F) -> Option<T> where T: Eq {
        if self.0.lock().is_some() {
            None
        } else {
            self.try_set_same(f())
        }
    }

    /// Tries to get a reference to the inner value, returns `None` if it is not yet initialized
    #[inline(always)]
    pub fn try_get(&self) -> Option<&T> {
        let lock = &*self.0.lock();
        if let Some(ref inner) = *lock {
            // This is safe since we won't mutate the inner value
            unsafe { Some(&*(inner as *const T)) }
        } else {
            None
        }
    }

    /// Gets reference to the inner value, panics if it is not yet initialized
    #[inline(always)]
    pub fn get(&self) -> &T {
        self.try_get().expect("value was not set")
    }

    /// Gets reference to the inner value, panics if it is not yet initialized
    #[inline(always)]
    pub fn borrow(&self) -> &T {
        self.get()
    }
}

#[derive(Debug)]
pub struct Lock<T>(InnerLock<T>);

impl<T> Lock<T> {
    #[inline(always)]
    pub fn new(inner: T) -> Self {
        Lock(InnerLock::new(inner))
    }

    #[inline(always)]
    pub fn into_inner(self) -> T {
        self.0.into_inner()
    }

    #[inline(always)]
    pub fn get_mut(&mut self) -> &mut T {
        self.0.get_mut()
    }

    #[cfg(parallel_compiler)]
    #[inline(always)]
    pub fn try_lock(&self) -> Option<LockGuard<'_, T>> {
        self.0.try_lock()
    }

    #[cfg(not(parallel_compiler))]
    #[inline(always)]
    pub fn try_lock(&self) -> Option<LockGuard<'_, T>> {
        self.0.try_borrow_mut().ok()
    }

    #[cfg(parallel_compiler)]
    #[inline(always)]
    pub fn lock(&self) -> LockGuard<'_, T> {
        if ERROR_CHECKING {
            self.0.try_lock().expect("lock was already held")
        } else {
            self.0.lock()
        }
    }

    #[cfg(not(parallel_compiler))]
    #[inline(always)]
    pub fn lock(&self) -> LockGuard<'_, T> {
        self.0.borrow_mut()
    }

    #[inline(always)]
    pub fn with_lock<F: FnOnce(&mut T) -> R, R>(&self, f: F) -> R {
        f(&mut *self.lock())
    }

    #[inline(always)]
    pub fn borrow(&self) -> LockGuard<'_, T> {
        self.lock()
    }

    #[inline(always)]
    pub fn borrow_mut(&self) -> LockGuard<'_, T> {
        self.lock()
    }
}

impl<T: Default> Default for Lock<T> {
    #[inline]
    fn default() -> Self {
        Lock::new(T::default())
    }
}

// FIXME: Probably a bad idea
impl<T: Clone> Clone for Lock<T> {
    #[inline]
    fn clone(&self) -> Self {
        Lock::new(self.borrow().clone())
    }
}

#[derive(Debug)]
pub struct RwLock<T>(InnerRwLock<T>);

impl<T> RwLock<T> {
    #[inline(always)]
    pub fn new(inner: T) -> Self {
        RwLock(InnerRwLock::new(inner))
    }

    #[inline(always)]
    pub fn into_inner(self) -> T {
        self.0.into_inner()
    }

    #[inline(always)]
    pub fn get_mut(&mut self) -> &mut T {
        self.0.get_mut()
    }

    #[cfg(not(parallel_compiler))]
    #[inline(always)]
    pub fn read(&self) -> ReadGuard<'_, T> {
        self.0.borrow()
    }

    #[cfg(parallel_compiler)]
    #[inline(always)]
    pub fn read(&self) -> ReadGuard<'_, T> {
        if ERROR_CHECKING {
            self.0.try_read().expect("lock was already held")
        } else {
            self.0.read()
        }
    }

    #[inline(always)]
    pub fn with_read_lock<F: FnOnce(&T) -> R, R>(&self, f: F) -> R {
        f(&*self.read())
    }

    #[cfg(not(parallel_compiler))]
    #[inline(always)]
    pub fn try_write(&self) -> Result<WriteGuard<'_, T>, ()> {
        self.0.try_borrow_mut().map_err(|_| ())
    }

    #[cfg(parallel_compiler)]
    #[inline(always)]
    pub fn try_write(&self) -> Result<WriteGuard<'_, T>, ()> {
        self.0.try_write().ok_or(())
    }

    #[cfg(not(parallel_compiler))]
    #[inline(always)]
    pub fn write(&self) -> WriteGuard<'_, T> {
        self.0.borrow_mut()
    }

    #[cfg(parallel_compiler)]
    #[inline(always)]
    pub fn write(&self) -> WriteGuard<'_, T> {
        if ERROR_CHECKING {
            self.0.try_write().expect("lock was already held")
        } else {
            self.0.write()
        }
    }

    #[inline(always)]
    pub fn with_write_lock<F: FnOnce(&mut T) -> R, R>(&self, f: F) -> R {
        f(&mut *self.write())
    }

    #[inline(always)]
    pub fn borrow(&self) -> ReadGuard<'_, T> {
        self.read()
    }

    #[inline(always)]
    pub fn borrow_mut(&self) -> WriteGuard<'_, T> {
        self.write()
    }
}

// FIXME: Probably a bad idea
impl<T: Clone> Clone for RwLock<T> {
    #[inline]
    fn clone(&self) -> Self {
        RwLock::new(self.borrow().clone())
    }
}

/// A type which only allows its inner value to be used in one thread.
/// It will panic if it is used on multiple threads.
#[derive(Copy, Clone, Hash, Debug, Eq, PartialEq)]
pub struct OneThread<T> {
    #[cfg(parallel_compiler)]
    thread: thread::ThreadId,
    inner: T,
}

#[cfg(parallel_compiler)]
unsafe impl<T> std::marker::Sync for OneThread<T> {}
#[cfg(parallel_compiler)]
unsafe impl<T> std::marker::Send for OneThread<T> {}

impl<T> OneThread<T> {
    #[inline(always)]
    fn check(&self) {
        #[cfg(parallel_compiler)]
        assert_eq!(thread::current().id(), self.thread);
    }

    #[inline(always)]
    pub fn new(inner: T) -> Self {
        OneThread {
            #[cfg(parallel_compiler)]
            thread: thread::current().id(),
            inner,
        }
    }

    #[inline(always)]
    pub fn into_inner(value: Self) -> T {
        value.check();
        value.inner
    }
}

impl<T> Deref for OneThread<T> {
    type Target = T;

    fn deref(&self) -> &T {
        self.check();
        &self.inner
    }
}

impl<T> DerefMut for OneThread<T> {
    fn deref_mut(&mut self) -> &mut T {
        self.check();
        &mut self.inner
    }
}
