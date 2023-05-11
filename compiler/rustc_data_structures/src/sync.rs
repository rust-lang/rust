//! This module defines various operations and types that are implemented in
//! one way for the serial compiler, and another way the parallel compiler.
//!
//! Operations
//! ----------
//! The parallel versions of operations use Rayon to execute code in parallel,
//! while the serial versions degenerate straightforwardly to serial execution.
//! The operations include `join`, `parallel`, `par_iter`, and `par_for_each`.
//!
//! Types
//! -----
//! The parallel versions of types provide various kinds of synchronization,
//! while the serial compiler versions do not.
//!
//! The following table shows how the types are implemented internally. Except
//! where noted otherwise, the type in column one is defined as a
//! newtype around the type from column two or three.
//!
//! | Type                    | Serial version      | Parallel version                |
//! | ----------------------- | ------------------- | ------------------------------- |
//! | `Lrc<T>`                | `rc::Rc<T>`         | `sync::Arc<T>`                  |
//! |` Weak<T>`               | `rc::Weak<T>`       | `sync::Weak<T>`                 |
//! |                         |                     |                                 |
//! | `AtomicBool`            | `Cell<bool>`        | `atomic::AtomicBool`            |
//! | `AtomicU32`             | `Cell<u32>`         | `atomic::AtomicU32`             |
//! | `AtomicU64`             | `Cell<u64>`         | `atomic::AtomicU64`             |
//! | `AtomicUsize`           | `Cell<usize>`       | `atomic::AtomicUsize`           |
//! |                         |                     |                                 |
//! | `Lock<T>`               | `RefCell<T>`        | `parking_lot::Mutex<T>`         |
//! | `RwLock<T>`             | `RefCell<T>`        | `parking_lot::RwLock<T>`        |
//! | `MTLock<T>`        [^1] | `T`                 | `Lock<T>`                       |
//! | `MTLockRef<'a, T>` [^2] | `&'a mut MTLock<T>` | `&'a MTLock<T>`                 |
//! |                         |                     |                                 |
//! | `ParallelIterator`      | `Iterator`          | `rayon::iter::ParallelIterator` |
//!
//! [^1] `MTLock` is similar to `Lock`, but the serial version avoids the cost
//! of a `RefCell`. This is appropriate when interior mutability is not
//! required.
//!
//! [^2] `MTLockRef` is a typedef.

pub use crate::marker::*;
use std::cell::{Cell, RefCell, RefMut, UnsafeCell};
use std::collections::HashMap;
use std::fmt::{Debug, Formatter};
use std::hash::{BuildHasher, Hash};
use std::intrinsics::likely;
use std::marker::PhantomData;
use std::ops::{Deref, DerefMut};
use std::panic::{catch_unwind, resume_unwind, AssertUnwindSafe};

mod worker_local;

pub use std::sync::atomic::Ordering;
pub use std::sync::atomic::Ordering::SeqCst;

pub use vec::{AppendOnlyIndexVec, AppendOnlyVec};

mod vec;
use parking_lot::lock_api::RawMutex as _;
use parking_lot::lock_api::RawRwLock as _;
use parking_lot::{Mutex, MutexGuard, RawMutex, RawRwLock};

mod mode {
    use super::Ordering;
    use std::sync::atomic::AtomicU8;

    const UNINITIALIZED: u8 = 0;
    const DYN_NOT_THREAD_SAFE: u8 = 1;
    const DYN_THREAD_SAFE: u8 = 2;

    static DYN_THREAD_SAFE_MODE: AtomicU8 = AtomicU8::new(UNINITIALIZED);

    // Whether thread safety is enabled (due to running under multiple threads).
    #[inline]
    pub fn is_dyn_thread_safe() -> bool {
        match DYN_THREAD_SAFE_MODE.load(Ordering::Relaxed) {
            DYN_NOT_THREAD_SAFE => false,
            DYN_THREAD_SAFE => true,
            _ => panic!("uninitialized dyn_thread_safe mode!"),
        }
    }

    // Only set by the `-Z threads` compile option
    pub fn set_dyn_thread_safe_mode(mode: bool) {
        let set: u8 = if mode { DYN_THREAD_SAFE } else { DYN_NOT_THREAD_SAFE };
        let previous = DYN_THREAD_SAFE_MODE.compare_exchange(
            UNINITIALIZED,
            set,
            Ordering::Relaxed,
            Ordering::Relaxed,
        );

        // Check that the mode was either uninitialized or was already set to the requested mode.
        assert!(previous.is_ok() || previous == Err(set));
    }
}

pub use mode::{is_dyn_thread_safe, set_dyn_thread_safe_mode};

cfg_if! {
    if #[cfg(not(parallel_compiler))] {
        pub unsafe auto trait Send {}
        pub unsafe auto trait Sync {}

        unsafe impl<T> Send for T {}
        unsafe impl<T> Sync for T {}

        use std::ops::Add;

        /// This is a single threaded variant of `AtomicU64`, `AtomicUsize`, etc.
        /// It has explicit ordering arguments and is only intended for use with
        /// the native atomic types.
        /// You should use this type through the `AtomicU64`, `AtomicUsize`, etc, type aliases
        /// as it's not intended to be used separately.
        #[derive(Debug, Default)]
        pub struct Atomic<T: Copy>(Cell<T>);

        impl<T: Copy> Atomic<T> {
            #[inline]
            pub fn new(v: T) -> Self {
                Atomic(Cell::new(v))
            }

            #[inline]
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

            #[inline]
            pub fn swap(&self, val: T, _: Ordering) -> T {
                self.0.replace(val)
            }
        }

        impl Atomic<bool> {
            pub fn fetch_or(&self, val: bool, _: Ordering) -> bool {
                let result = self.0.get() | val;
                self.0.set(val);
                result
            }
        }

        impl<T: Copy + PartialEq> Atomic<T> {
            #[inline]
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
            #[inline]
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

        pub fn join<A, B, RA, RB>(oper_a: A, oper_b: B) -> (RA, RB)
            where A: FnOnce() -> RA,
                  B: FnOnce() -> RB
        {
            (oper_a(), oper_b())
        }

        #[macro_export]
        macro_rules! parallel {
            ($($blocks:block),*) => {
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

        pub fn par_for_each_in<T: IntoIterator>(t: T, mut for_each: impl FnMut(T::Item) + Sync + Send) {
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

        pub fn par_map<T: IntoIterator, R, C: FromIterator<R>>(
            t: T,
            mut map: impl FnMut(<<T as IntoIterator>::IntoIter as Iterator>::Item) -> R,
        ) -> C {
            // We catch panics here ensuring that all the loop iterations execute.
            let mut panic = None;
            let r = t.into_iter().filter_map(|i| {
                match catch_unwind(AssertUnwindSafe(|| map(i))) {
                    Ok(r) => Some(r),
                    Err(p) => {
                        if panic.is_none() {
                            panic = Some(p);
                        }
                        None
                    }
                }
            }).collect();
            if let Some(panic) = panic {
                resume_unwind(panic);
            }
            r
        }

        pub use std::rc::Rc as Lrc;
        pub use std::rc::Weak as Weak;

        pub use std::cell::OnceCell;

        pub type MTLockRef<'a, T> = &'a mut MTLock<T>;

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

        pub use std::sync::OnceLock as OnceCell;

        pub use std::sync::atomic::{AtomicBool, AtomicUsize, AtomicU32, AtomicU64};

        pub use std::sync::Arc as Lrc;
        pub use std::sync::Weak as Weak;

        pub type MTLockRef<'a, T> = &'a MTLock<T>;

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

        #[inline]
        pub fn join<A, B, RA: DynSend, RB: DynSend>(oper_a: A, oper_b: B) -> (RA, RB)
        where
            A: FnOnce() -> RA + DynSend,
            B: FnOnce() -> RB + DynSend,
        {
            if mode::active() {
                let oper_a = FromDyn::from(oper_a);
                let oper_b = FromDyn::from(oper_b);
                let (a, b) = rayon::join(move || FromDyn::from(oper_a.into_inner()()), move || FromDyn::from(oper_b.into_inner()()));
                (a.into_inner(), b.into_inner())
            } else {
                (oper_a(), oper_b())
            }
        }

        use std::thread;

        #[inline]
        pub fn join<A, B, RA: DynSend, RB: DynSend>(oper_a: A, oper_b: B) -> (RA, RB)
        where
            A: FnOnce() -> RA + DynSend,
            B: FnOnce() -> RB + DynSend,
        {
            if mode::is_dyn_thread_safe() {
                let oper_a = FromDyn::from(oper_a);
                let oper_b = FromDyn::from(oper_b);
                let (a, b) = rayon::join(move || FromDyn::from(oper_a.into_inner()()), move || FromDyn::from(oper_b.into_inner()()));
                (a.into_inner(), b.into_inner())
            } else {
                (oper_a(), oper_b())
            }
        }

        // This function only works when `mode::is_dyn_thread_safe()`.
        pub fn scope<'scope, OP, R>(op: OP) -> R
        where
            OP: FnOnce(&rayon::Scope<'scope>) -> R + DynSend,
            R: DynSend,
        {
            let op = FromDyn::from(op);
            rayon::scope(|s| FromDyn::from(op.into_inner()(s))).into_inner()
        }

        /// Runs a list of blocks in parallel. The first block is executed immediately on
        /// the current thread. Use that for the longest running block.
        #[macro_export]
        macro_rules! parallel {
            (impl $fblock:tt [$($c:tt,)*] [$block:tt $(, $rest:tt)*]) => {
            (impl $fblock:block [$($c:expr,)*] [$block:expr $(, $rest:expr)*]) => {
                parallel!(impl $fblock [$block, $($c,)*] [$($rest),*])
            };
            (impl $fblock:tt [$($blocks:tt,)*] []) => {
            (impl $fblock:block [$($blocks:expr,)*] []) => {
                ::rustc_data_structures::sync::scope(|s| {






                    $(let block = rustc_data_structures::sync::FromDyn::from(|| $blocks);
                    s.spawn(move |_| block.into_inner()());)*
                    (|| $fblock)();
                });
            };
            ($fblock:block, $($blocks:block),*) => {
                if rustc_data_structures::sync::is_dyn_thread_safe() {
                    // Reverse the order of the later blocks since Rayon executes them in reverse order
                    // when using a single thread. This ensures the execution order matches that
                    // of a single threaded rustc.
                    parallel!(impl $fblock [] [$($blocks),*]);
                } else {
                    // We catch panics here ensuring that all the blocks execute.
                    // This makes behavior consistent with the parallel compiler.
                    let mut panic = None;
                    if let Err(p) = ::std::panic::catch_unwind(
                        ::std::panic::AssertUnwindSafe(|| $fblock)
                    ) {
                        if panic.is_none() {
                            panic = Some(p);
                        }
                    }
                    $(
                        s.spawn(|_| $blocks);
                        if let Err(p) = ::std::panic::catch_unwind(
                            ::std::panic::AssertUnwindSafe(|| $blocks)
                        ) {
                            if panic.is_none() {
                                panic = Some(p);
                            }
                        }
                    )*
                    $fblock;
                })
                    if let Some(panic) = panic {
                        ::std::panic::resume_unwind(panic);
                    }
                }
            };
            ($fblock:tt, $($blocks:tt),*) => {
                // Reverse the order of the later blocks since Rayon executes them in reverse order
                // when using a single thread. This ensures the execution order matches that
                // of a single threaded rustc
                parallel!(impl $fblock [] [$($blocks),*]);
            };
        }

        use rayon::iter::{FromParallelIterator, IntoParallelIterator, ParallelIterator};

        pub fn par_for_each_in<I, T: IntoIterator<Item = I> + IntoParallelIterator<Item = I>>(
            t: T,
            for_each: impl Fn(I) + DynSync + DynSend
        ) {
            if mode::is_dyn_thread_safe() {
                let for_each = FromDyn::from(for_each);
                let panic: Lock<Option<_>> = Lock::new(None);
                t.into_par_iter().for_each(|i| if let Err(p) = catch_unwind(AssertUnwindSafe(|| for_each(i))) {
                    let mut l = panic.lock();
                    if l.is_none() {
                        *l = Some(p)
                    }
                });

                if let Some(panic) = panic.into_inner() {
                    resume_unwind(panic);
                }
            } else {
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
        }

        pub fn par_map<
            I,
            T: IntoIterator<Item = I> + IntoParallelIterator<Item = I>,
            R: std::marker::Send,
            C: FromIterator<R> + FromParallelIterator<R>
        >(
            t: T,
            map: impl Fn(I) -> R + DynSync + DynSend
        ) -> C {
            if mode::is_dyn_thread_safe() {
                let panic: Lock<Option<_>> = Lock::new(None);
                let map = FromDyn::from(map);
                // We catch panics here ensuring that all the loop iterations execute.
                let r = t.into_par_iter().filter_map(|i| {
                    match catch_unwind(AssertUnwindSafe(|| map(i))) {
                        Ok(r) => Some(r),
                        Err(p) => {
                            let mut l = panic.lock();
                            if l.is_none() {
                                *l = Some(p);
                            }
                            None
                        },
                    }
                }).collect();

                if let Some(panic) = panic.into_inner() {
                    resume_unwind(panic);
                }
                r
            } else {
                // We catch panics here ensuring that all the loop iterations execute.
                let mut panic = None;
                let r = t.into_iter().filter_map(|i| {
                    match catch_unwind(AssertUnwindSafe(|| map(i))) {
                        Ok(r) => Some(r),
                        Err(p) => {
                            if panic.is_none() {
                                panic = Some(p);
                            }
                            None
                        }
                    }
                }).collect();
                if let Some(panic) = panic {
                    resume_unwind(panic);
                }
                r
            }
        }

pub unsafe trait DynSend {}
pub unsafe trait DynSync {}

unsafe impl<T> DynSend for T where T: Send {}
unsafe impl<T> DynSync for T where T: Sync {}

#[derive(Copy, Clone)]
pub struct FromDyn<T>(T);

impl<T> FromDyn<T> {
    #[inline(always)]
    pub fn from(val: T) -> Self {
        // Check that `sync::active()` is true on creation so we can
        // implement `Send` and `Sync` for this structure when `T`
        // implements `DynSend` and `DynSync` respectively.
        #[cfg(parallel_compiler)]
        assert!(mode::active());
        FromDyn(val)
    }

    #[inline(always)]
    pub fn into_inner(self) -> T {
        self.0
    }
}

#[derive(Default, Debug)]
#[repr(align(64))]
pub struct CacheAligned<T>(pub T);

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

/// This makes locks panic if they are already held.
/// It is only useful when you are running in a single thread
// const ERROR_CHECKING: bool = false;

pub struct Lock<T> {
    single_thread: bool,
    pub(crate) data: UnsafeCell<T>,
    pub(crate) borrow: Cell<bool>,
    mutex: RawMutex,
}

impl<T: Debug> Debug for Lock<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self.try_lock() {
            Some(guard) => f.debug_struct("Lock").field("data", guard.deref()).finish(),
            None => {
                struct LockedPlaceholder;
                impl Debug for LockedPlaceholder {
                    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
                        f.write_str("<locked>")
                    }
                }

                f.debug_struct("Lock").field("data", &LockedPlaceholder).finish()
            }
        }
    }
}

impl<T> Lock<T> {
    #[inline]
    pub fn new(val: T) -> Self {
        Lock {
            single_thread: !active(),
            data: UnsafeCell::new(val),
            borrow: Cell::new(false),
            mutex: RawMutex::INIT,
        }
    }

    #[inline]
    pub fn into_inner(self) -> T {
        self.data.into_inner()
    }

    #[inline]
    pub fn get_mut(&mut self) -> &mut T {
        self.data.get_mut()
    }

    #[inline]
    pub fn try_lock(&self) -> Option<LockGuard<'_, T>> {
        // SAFETY: the `&mut T` is accessible as long as self exists.
        if likely(self.single_thread) {
            if self.borrow.get() {
                None
            } else {
                self.borrow.set(true);
                Some(LockGuard { lock: &self, marker: PhantomData })
            }
        } else {
            if !self.mutex.try_lock() {
                None
            } else {
                Some(LockGuard { lock: &self, marker: PhantomData })
            }
        }
    }

    #[inline]
    fn lock_raw(&self) {
        if likely(self.single_thread) {
            assert!(!self.borrow.replace(true));
        } else {
            self.mutex.lock();
        }
    }

    #[inline(always)]
    #[track_caller]
    pub fn lock(&self) -> LockGuard<'_, T> {
        self.lock_raw();
        LockGuard { lock: &self, marker: PhantomData }
    }

    #[inline]
    pub(crate) fn with_mt_lock<F: FnOnce(&mut T) -> R, R>(&self, f: F) -> R {
        unsafe {
            self.mutex.lock();
            let r = f(&mut *self.data.get());
            self.mutex.unlock();
            r
        }
    }

    #[inline(always)]
    #[track_caller]
    pub fn with_lock<F: FnOnce(&mut T) -> R, R>(&self, f: F) -> R {
        if likely(self.single_thread) {
            assert!(!self.borrow.replace(true));
            let r = unsafe { f(&mut *self.data.get()) };
            self.borrow.set(false);
            r
        } else {
            self.with_mt_lock(f)
        }
    }

    #[inline]
    fn with_mt_borrow<F: FnOnce(&T) -> R, R>(&self, f: F) -> R {
        unsafe {
            self.mutex.lock();
            let r = f(&*self.data.get());
            self.mutex.unlock();
            r
        }
    }

    #[inline(always)]
    #[track_caller]
    pub fn with_borrow<F: FnOnce(&T) -> R, R>(&self, f: F) -> R {
        if likely(self.single_thread) {
            assert!(!self.borrow.replace(true));
            let r = unsafe { f(&*self.data.get()) };
            self.borrow.set(false);
            r
        } else {
            self.with_mt_borrow(f)
        }
    }

    #[inline(always)]
    #[track_caller]
    pub fn borrow(&self) -> LockGuard<'_, T> {
        self.lock()
    }

    #[inline(always)]
    #[track_caller]
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

// Just for speed test
unsafe impl<T: Send> std::marker::Send for Lock<T> {}
unsafe impl<T: Send> std::marker::Sync for Lock<T> {}

pub struct LockGuard<'a, T> {
    lock: &'a Lock<T>,
    marker: PhantomData<&'a mut T>,
}

impl<T> Deref for LockGuard<'_, T> {
    type Target = T;

    #[inline(always)]
    fn deref(&self) -> &T {
        unsafe { &*self.lock.data.get() }
    }
}

impl<T> DerefMut for LockGuard<'_, T> {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut T {
        unsafe { &mut *self.lock.data.get() }
    }
}

#[inline]
fn unlock_mt<T>(guard: &mut LockGuard<'_, T>) {
    unsafe { guard.lock.mutex.unlock() }
}

impl<'a, T> Drop for LockGuard<'a, T> {
    #[inline]
    fn drop(&mut self) {
        if likely(self.lock.single_thread) {
            debug_assert!(self.lock.borrow.get());
            self.lock.borrow.set(false);
        } else {
            unlock_mt(self)
        }
    }
}

pub trait SLock: Copy {
    type Lock<T>: LockLike<T>;
}

pub trait LockLike<T> {
    type LockGuard<'a>: DerefMut<Target = T>
    where
        Self: 'a;

    fn new(val: T) -> Self;

    fn into_inner(self) -> T;

    fn get_mut(&mut self) -> &mut T;

    fn try_lock(&self) -> Option<Self::LockGuard<'_>>;

    fn lock(&self) -> Self::LockGuard<'_>;
}

#[derive(Copy, Clone, Default)]
pub struct SRefCell;

impl SLock for SRefCell {
    type Lock<T> = RefCell<T>;
}

impl<T> LockLike<T> for RefCell<T> {
    type LockGuard<'a> = RefMut<'a, T> where T: 'a;

    #[inline]
    fn new(val: T) -> Self {
        RefCell::new(val)
    }

    #[inline]
    fn into_inner(self) -> T {
        self.into_inner()
    }

    #[inline]
    fn get_mut(&mut self) -> &mut T {
        self.get_mut()
    }

    #[inline]
    fn try_lock(&self) -> Option<RefMut<'_, T>> {
        self.try_borrow_mut().ok()
    }

    #[inline(always)]
    #[track_caller]
    fn lock(&self) -> RefMut<'_, T> {
        self.borrow_mut()
    }
}

#[derive(Copy, Clone, Default)]
pub struct SMutex;

impl SLock for SMutex {
    type Lock<T> = Mutex<T>;
}

impl<T> LockLike<T> for Mutex<T> {
    type LockGuard<'a> = MutexGuard<'a, T> where T: 'a;

    #[inline]
    fn new(val: T) -> Self {
        Mutex::new(val)
    }

    #[inline]
    fn into_inner(self) -> T {
        self.into_inner()
    }

    #[inline]
    fn get_mut(&mut self) -> &mut T {
        self.get_mut()
    }

    #[inline]
    fn try_lock(&self) -> Option<MutexGuard<'_, T>> {
        self.try_lock()
    }

    #[inline(always)]
    #[track_caller]
    fn lock(&self) -> MutexGuard<'_, T> {
        self.lock()
    }
}

pub struct MappedReadGuard<'a, T: ?Sized> {
    raw: &'a RwLockRaw,
    data: *const T,
    marker: PhantomData<&'a T>,
}

unsafe impl<T: ?Sized + Sync> std::marker::Send for MappedReadGuard<'_, T> {}
unsafe impl<T: ?Sized + Sync> std::marker::Sync for MappedReadGuard<'_, T> {}

impl<'a, T: 'a + ?Sized> MappedReadGuard<'a, T> {
    #[inline]
    pub fn map<U: ?Sized, F>(s: Self, f: F) -> MappedReadGuard<'a, U>
    where
        F: FnOnce(&T) -> &U,
    {
        let raw = s.raw;
        let data = f(unsafe { &*s.data });
        std::mem::forget(s);
        MappedReadGuard { raw, data, marker: PhantomData }
    }
}

impl<'a, T: 'a + ?Sized> Deref for MappedReadGuard<'a, T> {
    type Target = T;
    #[inline]
    fn deref(&self) -> &T {
        unsafe { &*self.data }
    }
}

impl<'a, T: 'a + ?Sized> Drop for MappedReadGuard<'a, T> {
    #[inline]
    fn drop(&mut self) {
        if likely(self.raw.single_thread) {
            let i = self.raw.borrow.get();
            debug_assert!(i > 0);
            self.raw.borrow.set(i - 1);
        } else {
            // Safety: An RwLockReadGuard always holds a shared lock.
            unsafe {
                self.raw.raw.unlock_shared();
            }
        }
    }
}

pub struct MappedWriteGuard<'a, T: ?Sized> {
    raw: &'a RwLockRaw,
    data: *mut T,
    marker: PhantomData<&'a mut T>,
}

unsafe impl<T: ?Sized + Sync> std::marker::Send for MappedWriteGuard<'_, T> {}

impl<'a, T: 'a + ?Sized> MappedWriteGuard<'a, T> {
    #[inline]
    pub fn map<U: ?Sized, F>(s: Self, f: F) -> MappedWriteGuard<'a, U>
    where
        F: FnOnce(&mut T) -> &mut U,
    {
        let raw = s.raw;
        let data = f(unsafe { &mut *s.data });
        std::mem::forget(s);
        MappedWriteGuard { raw, data, marker: PhantomData }
    }
}

impl<'a, T: 'a + ?Sized> Deref for MappedWriteGuard<'a, T> {
    type Target = T;
    #[inline]
    fn deref(&self) -> &T {
        unsafe { &*self.data }
    }
}

impl<'a, T: 'a + ?Sized> DerefMut for MappedWriteGuard<'a, T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut T {
        unsafe { &mut *self.data }
    }
}

impl<'a, T: 'a + ?Sized> Drop for MappedWriteGuard<'a, T> {
    #[inline]
    fn drop(&mut self) {
        if likely(self.raw.single_thread) {
            assert_eq!(self.raw.borrow.replace(0), -1);
        } else {
            // Safety: An RwLockReadGuard always holds a shared lock.
            unsafe {
                self.raw.raw.unlock_exclusive();
            }
        }
    }
}

pub struct ReadGuard<'a, T> {
    rwlock: &'a RwLock<T>,
    marker: PhantomData<&'a T>,
}

impl<'a, T: 'a> ReadGuard<'a, T> {
    pub fn map<U: ?Sized, F>(s: Self, f: F) -> MappedReadGuard<'a, U>
    where
        F: FnOnce(&T) -> &U,
    {
        let raw = &s.rwlock.raw;
        let data = f(unsafe { &*s.rwlock.data.get() });
        std::mem::forget(s);
        MappedReadGuard { raw, data, marker: PhantomData }
    }
}

impl<'a, T: 'a> Deref for ReadGuard<'a, T> {
    type Target = T;
    #[inline]
    fn deref(&self) -> &T {
        unsafe { &*self.rwlock.data.get() }
    }
}

impl<'a, T: 'a> Drop for ReadGuard<'a, T> {
    #[inline]
    fn drop(&mut self) {
        if likely(self.rwlock.raw.single_thread) {
            let i = self.rwlock.raw.borrow.get();
            debug_assert!(i > 0);
            self.rwlock.raw.borrow.set(i - 1);
        } else {
            // Safety: An RwLockReadGuard always holds a shared lock.
            unsafe {
                self.rwlock.raw.raw.unlock_shared();
            }
        }
    }
}

pub struct WriteGuard<'a, T> {
    rwlock: &'a RwLock<T>,
    marker: PhantomData<&'a mut T>,
}

impl<'a, T: 'a> WriteGuard<'a, T> {
    pub fn map<U: ?Sized, F>(s: Self, f: F) -> MappedWriteGuard<'a, U>
    where
        F: FnOnce(&mut T) -> &mut U,
    {
        let raw = &s.rwlock.raw;
        let data = f(unsafe { &mut *s.rwlock.data.get() });
        std::mem::forget(s);
        MappedWriteGuard { raw, data, marker: PhantomData }
    }
}

impl<'a, T: 'a> Deref for WriteGuard<'a, T> {
    type Target = T;
    #[inline]
    fn deref(&self) -> &T {
        unsafe { &*self.rwlock.data.get() }
    }
}

impl<'a, T: 'a> DerefMut for WriteGuard<'a, T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut T {
        unsafe { &mut *self.rwlock.data.get() }
    }
}

impl<'a, T: 'a> Drop for WriteGuard<'a, T> {
    #[inline]
    fn drop(&mut self) {
        if likely(self.rwlock.raw.single_thread) {
            assert_eq!(self.rwlock.raw.borrow.replace(0), -1);
        } else {
            // Safety: An RwLockWriteGuard always holds an exclusive lock.
            unsafe {
                self.rwlock.raw.raw.unlock_exclusive();
            }
        }
    }
}

struct RwLockRaw {
    single_thread: bool,
    borrow: Cell<isize>,
    raw: RawRwLock,
}

pub struct RwLock<T> {
    raw: RwLockRaw,
    data: UnsafeCell<T>,
}

impl<T: Debug> Debug for RwLock<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Lock").field("data", self.read().deref()).finish()
    }
}

impl<T: Default> Default for RwLock<T> {
    fn default() -> Self {
        RwLock {
            raw: RwLockRaw { single_thread: !active(), borrow: Cell::new(0), raw: RawRwLock::INIT },

            data: UnsafeCell::new(T::default()),
        }
    }
}

impl<T> RwLock<T> {
    #[inline(always)]
    pub fn new(inner: T) -> Self {
        RwLock {
            raw: RwLockRaw { single_thread: !active(), borrow: Cell::new(0), raw: RawRwLock::INIT },

            data: UnsafeCell::new(inner),
        }
    }

    #[inline(always)]
    pub fn into_inner(self) -> T {
        self.data.into_inner()
    }

    #[inline(always)]
    pub fn get_mut(&mut self) -> &mut T {
        self.data.get_mut()
    }

    #[inline]
    fn mt_read(&self) -> ReadGuard<'_, T> {
        self.raw.raw.lock_shared();
        ReadGuard { rwlock: self, marker: PhantomData }
    }

    #[inline(always)]
    pub fn read(&self) -> ReadGuard<'_, T> {
        if likely(self.raw.single_thread) {
            let b = self.raw.borrow.get();
            assert!(b >= 0);
            self.raw.borrow.set(b + 1);
            ReadGuard { rwlock: self, marker: PhantomData }
        } else {
            self.mt_read()
        }
    }

    #[inline]
    fn with_mt_read_lock<F: FnOnce(&T) -> R, R>(&self, f: F) -> R {
        self.raw.raw.lock_shared();
        let r = unsafe { f(&*self.data.get()) };
        unsafe {
            self.raw.raw.unlock_shared();
        }
        r
    }

    #[inline(always)]
    #[track_caller]
    pub fn with_read_lock<F: FnOnce(&T) -> R, R>(&self, f: F) -> R {
        if likely(self.raw.single_thread) {
            let b = self.raw.borrow.get();
            assert!(b >= 0);
            self.raw.borrow.set(b + 1);
            let r = unsafe { f(&*self.data.get()) };
            self.raw.borrow.set(b);
            r
        } else {
            self.with_mt_read_lock(f)
        }
    }

    #[inline(always)]
    pub fn try_write(&self) -> Result<WriteGuard<'_, T>, ()> {
        if likely(self.raw.single_thread) {
            let b = self.raw.borrow.get();
            if b != 0 {
                Err(())
            } else {
                self.raw.borrow.set(-1);
                Ok(WriteGuard { rwlock: self, marker: PhantomData })
            }
        } else {
            if self.raw.raw.try_lock_exclusive() {
                Ok(WriteGuard { rwlock: self, marker: PhantomData })
            } else {
                Err(())
            }
        }
    }

    #[inline]
    fn mt_write(&self) -> WriteGuard<'_, T> {
        self.raw.raw.lock_exclusive();
        WriteGuard { rwlock: self, marker: PhantomData }
    }

    #[inline(always)]
    pub fn write(&self) -> WriteGuard<'_, T> {
        if likely(self.raw.single_thread) {
            assert_eq!(self.raw.borrow.replace(-1), 0);
            WriteGuard { rwlock: self, marker: PhantomData }
        } else {
            self.mt_write()
        }
    }

    #[inline]
    pub fn with_mt_write_lock<F: FnOnce(&mut T) -> R, R>(&self, f: F) -> R {
        self.raw.raw.lock_exclusive();
        unsafe {
            let r = f(&mut *self.data.get());
            self.raw.raw.unlock_exclusive();
            r
        }
    }

    #[inline(always)]
    #[track_caller]
    pub fn with_write_lock<F: FnOnce(&mut T) -> R, R>(&self, f: F) -> R {
        if likely(self.raw.single_thread) {
            let b = self.raw.borrow.get();
            assert!(b >= 0);
            self.raw.borrow.set(b + 1);
            let r = unsafe { f(&mut *self.data.get()) };
            self.raw.borrow.set(b);
            r
        } else {
            self.with_mt_write_lock(f)
        }
    }

    #[inline(always)]
    #[track_caller]
    pub fn borrow(&self) -> ReadGuard<'_, T> {
        self.read()
    }

    #[inline(always)]
    #[track_caller]
    pub fn borrow_mut(&self) -> WriteGuard<'_, T> {
        self.write()
    }

    #[inline(always)]
    pub fn leak(&self) -> &T {
        let guard = self.read();
        let ret = unsafe { &*(&*guard as *const T) };
        std::mem::forget(guard);
        ret
    }
}

// just for speed test
unsafe impl<T: Send> std::marker::Send for RwLock<T> {}
unsafe impl<T: Send + Sync> std::marker::Sync for RwLock<T> {}

// FIXME: Probably a bad idea
impl<T: Clone> Clone for RwLock<T> {
    #[inline]
    fn clone(&self) -> Self {
        RwLock::new(self.borrow().clone())
    }
}

#[derive(Debug)]
pub struct WorkerLocal<T> {
    single_thread: bool,
    inner: Option<T>,
    mt_inner: Option<worker_local::WorkerLocal<T>>,
}

impl<T> WorkerLocal<T> {
    /// Creates a new worker local where the `initial` closure computes the
    /// value this worker local should take for each thread in the thread pool.
    #[inline]
    pub fn new<F: FnMut(usize) -> T>(mut f: F) -> WorkerLocal<T> {
        if !active() {
            WorkerLocal { single_thread: true, inner: Some(f(0)), mt_inner: None }
        } else {
            WorkerLocal {
                single_thread: false,
                inner: None,
                mt_inner: Some(worker_local::WorkerLocal::new(f)),
            }
        }
    }
}

impl<T> Deref for WorkerLocal<T> {
    type Target = T;

    #[inline(always)]
    fn deref(&self) -> &T {
        if self.single_thread {
            self.inner.as_ref().unwrap()
        } else {
            self.mt_inner.as_ref().unwrap().deref()
        }
    }
}

// Just for speed test
unsafe impl<T: Send> std::marker::Sync for WorkerLocal<T> {}

use std::thread;
pub use worker_local::Registry;

/// A type which only allows its inner value to be used in one thread.
/// It will panic if it is used on multiple threads.
#[derive(Debug)]
pub struct OneThread<T> {
    single_thread: bool,
    thread: thread::ThreadId,
    inner: T,
}

// just for speed test now
unsafe impl<T> std::marker::Sync for OneThread<T> {}
unsafe impl<T> std::marker::Send for OneThread<T> {}

impl<T> OneThread<T> {
    #[inline(always)]
    fn check(&self) {
        assert!(self.single_thread || thread::current().id() == self.thread);
    }

    #[inline(always)]
    pub fn new(inner: T) -> Self {
        OneThread { single_thread: !active(), thread: thread::current().id(), inner }
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
