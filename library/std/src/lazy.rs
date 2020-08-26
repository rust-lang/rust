//! Lazy values and one-time initialization of static data.

use crate::{
    cell::{Cell, UnsafeCell},
    fmt,
    mem::{self, MaybeUninit},
    ops::{Deref, Drop},
    panic::{RefUnwindSafe, UnwindSafe},
    sync::Once,
};

#[doc(inline)]
#[unstable(feature = "once_cell", issue = "74465")]
pub use core::lazy::*;

/// A synchronization primitive which can be written to only once.
///
/// This type is a thread-safe `OnceCell`.
///
/// # Examples
///
/// ```
/// #![feature(once_cell)]
///
/// use std::lazy::SyncOnceCell;
///
/// static CELL: SyncOnceCell<String> = SyncOnceCell::new();
/// assert!(CELL.get().is_none());
///
/// std::thread::spawn(|| {
///     let value: &String = CELL.get_or_init(|| {
///         "Hello, World!".to_string()
///     });
///     assert_eq!(value, "Hello, World!");
/// }).join().unwrap();
///
/// let value: Option<&String> = CELL.get();
/// assert!(value.is_some());
/// assert_eq!(value.unwrap().as_str(), "Hello, World!");
/// ```
#[unstable(feature = "once_cell", issue = "74465")]
pub struct SyncOnceCell<T> {
    once: Once,
    // Whether or not the value is initialized is tracked by `state_and_queue`.
    value: UnsafeCell<MaybeUninit<T>>,
}

// Why do we need `T: Send`?
// Thread A creates a `SyncOnceCell` and shares it with
// scoped thread B, which fills the cell, which is
// then destroyed by A. That is, destructor observes
// a sent value.
#[unstable(feature = "once_cell", issue = "74465")]
unsafe impl<T: Sync + Send> Sync for SyncOnceCell<T> {}
#[unstable(feature = "once_cell", issue = "74465")]
unsafe impl<T: Send> Send for SyncOnceCell<T> {}

#[unstable(feature = "once_cell", issue = "74465")]
impl<T: RefUnwindSafe + UnwindSafe> RefUnwindSafe for SyncOnceCell<T> {}
#[unstable(feature = "once_cell", issue = "74465")]
impl<T: UnwindSafe> UnwindSafe for SyncOnceCell<T> {}

#[unstable(feature = "once_cell", issue = "74465")]
impl<T> Default for SyncOnceCell<T> {
    fn default() -> SyncOnceCell<T> {
        SyncOnceCell::new()
    }
}

#[unstable(feature = "once_cell", issue = "74465")]
impl<T: fmt::Debug> fmt::Debug for SyncOnceCell<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.get() {
            Some(v) => f.debug_tuple("Once").field(v).finish(),
            None => f.write_str("Once(Uninit)"),
        }
    }
}

#[unstable(feature = "once_cell", issue = "74465")]
impl<T: Clone> Clone for SyncOnceCell<T> {
    fn clone(&self) -> SyncOnceCell<T> {
        let cell = Self::new();
        if let Some(value) = self.get() {
            match cell.set(value.clone()) {
                Ok(()) => (),
                Err(_) => unreachable!(),
            }
        }
        cell
    }
}

#[unstable(feature = "once_cell", issue = "74465")]
impl<T> From<T> for SyncOnceCell<T> {
    fn from(value: T) -> Self {
        let cell = Self::new();
        match cell.set(value) {
            Ok(()) => cell,
            Err(_) => unreachable!(),
        }
    }
}

#[unstable(feature = "once_cell", issue = "74465")]
impl<T: PartialEq> PartialEq for SyncOnceCell<T> {
    fn eq(&self, other: &SyncOnceCell<T>) -> bool {
        self.get() == other.get()
    }
}

#[unstable(feature = "once_cell", issue = "74465")]
impl<T: Eq> Eq for SyncOnceCell<T> {}

impl<T> SyncOnceCell<T> {
    /// Creates a new empty cell.
    #[unstable(feature = "once_cell", issue = "74465")]
    pub const fn new() -> SyncOnceCell<T> {
        SyncOnceCell { once: Once::new(), value: UnsafeCell::new(MaybeUninit::uninit()) }
    }

    /// Gets the reference to the underlying value.
    ///
    /// Returns `None` if the cell is empty, or being initialized. This
    /// method never blocks.
    #[unstable(feature = "once_cell", issue = "74465")]
    pub fn get(&self) -> Option<&T> {
        if self.is_initialized() {
            // Safe b/c checked is_initialized
            Some(unsafe { self.get_unchecked() })
        } else {
            None
        }
    }

    /// Gets the mutable reference to the underlying value.
    ///
    /// Returns `None` if the cell is empty. This method never blocks.
    #[unstable(feature = "once_cell", issue = "74465")]
    pub fn get_mut(&mut self) -> Option<&mut T> {
        if self.is_initialized() {
            // Safe b/c checked is_initialized and we have a unique access
            Some(unsafe { self.get_unchecked_mut() })
        } else {
            None
        }
    }

    /// Sets the contents of this cell to `value`.
    ///
    /// Returns `Ok(())` if the cell's value was updated.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(once_cell)]
    ///
    /// use std::lazy::SyncOnceCell;
    ///
    /// static CELL: SyncOnceCell<i32> = SyncOnceCell::new();
    ///
    /// fn main() {
    ///     assert!(CELL.get().is_none());
    ///
    ///     std::thread::spawn(|| {
    ///         assert_eq!(CELL.set(92), Ok(()));
    ///     }).join().unwrap();
    ///
    ///     assert_eq!(CELL.set(62), Err(62));
    ///     assert_eq!(CELL.get(), Some(&92));
    /// }
    /// ```
    #[unstable(feature = "once_cell", issue = "74465")]
    pub fn set(&self, value: T) -> Result<(), T> {
        let mut value = Some(value);
        self.get_or_init(|| value.take().unwrap());
        match value {
            None => Ok(()),
            Some(value) => Err(value),
        }
    }

    /// Gets the contents of the cell, initializing it with `f` if the cell
    /// was empty.
    ///
    /// Many threads may call `get_or_init` concurrently with different
    /// initializing functions, but it is guaranteed that only one function
    /// will be executed.
    ///
    /// # Panics
    ///
    /// If `f` panics, the panic is propagated to the caller, and the cell
    /// remains uninitialized.
    ///
    /// It is an error to reentrantly initialize the cell from `f`. The
    /// exact outcome is unspecified. Current implementation deadlocks, but
    /// this may be changed to a panic in the future.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(once_cell)]
    ///
    /// use std::lazy::SyncOnceCell;
    ///
    /// let cell = SyncOnceCell::new();
    /// let value = cell.get_or_init(|| 92);
    /// assert_eq!(value, &92);
    /// let value = cell.get_or_init(|| unreachable!());
    /// assert_eq!(value, &92);
    /// ```
    #[unstable(feature = "once_cell", issue = "74465")]
    pub fn get_or_init<F>(&self, f: F) -> &T
    where
        F: FnOnce() -> T,
    {
        match self.get_or_try_init(|| Ok::<T, !>(f())) {
            Ok(val) => val,
        }
    }

    /// Gets the contents of the cell, initializing it with `f` if
    /// the cell was empty. If the cell was empty and `f` failed, an
    /// error is returned.
    ///
    /// # Panics
    ///
    /// If `f` panics, the panic is propagated to the caller, and
    /// the cell remains uninitialized.
    ///
    /// It is an error to reentrantly initialize the cell from `f`.
    /// The exact outcome is unspecified. Current implementation
    /// deadlocks, but this may be changed to a panic in the future.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(once_cell)]
    ///
    /// use std::lazy::SyncOnceCell;
    ///
    /// let cell = SyncOnceCell::new();
    /// assert_eq!(cell.get_or_try_init(|| Err(())), Err(()));
    /// assert!(cell.get().is_none());
    /// let value = cell.get_or_try_init(|| -> Result<i32, ()> {
    ///     Ok(92)
    /// });
    /// assert_eq!(value, Ok(&92));
    /// assert_eq!(cell.get(), Some(&92))
    /// ```
    #[unstable(feature = "once_cell", issue = "74465")]
    pub fn get_or_try_init<F, E>(&self, f: F) -> Result<&T, E>
    where
        F: FnOnce() -> Result<T, E>,
    {
        // Fast path check
        // NOTE: We need to perform an acquire on the state in this method
        // in order to correctly synchronize `SyncLazy::force`. This is
        // currently done by calling `self.get()`, which in turn calls
        // `self.is_initialized()`, which in turn performs the acquire.
        if let Some(value) = self.get() {
            return Ok(value);
        }
        self.initialize(f)?;

        debug_assert!(self.is_initialized());

        // Safety: The inner value has been initialized
        Ok(unsafe { self.get_unchecked() })
    }

    /// Consumes the `SyncOnceCell`, returning the wrapped value. Returns
    /// `None` if the cell was empty.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(once_cell)]
    ///
    /// use std::lazy::SyncOnceCell;
    ///
    /// let cell: SyncOnceCell<String> = SyncOnceCell::new();
    /// assert_eq!(cell.into_inner(), None);
    ///
    /// let cell = SyncOnceCell::new();
    /// cell.set("hello".to_string()).unwrap();
    /// assert_eq!(cell.into_inner(), Some("hello".to_string()));
    /// ```
    #[unstable(feature = "once_cell", issue = "74465")]
    pub fn into_inner(mut self) -> Option<T> {
        // Safety: Safe because we immediately free `self` without dropping
        let inner = unsafe { self.take_inner() };

        // Don't drop this `SyncOnceCell`. We just moved out one of the fields, but didn't set
        // the state to uninitialized.
        mem::forget(self);
        inner
    }

    /// Takes the value out of this `SyncOnceCell`, moving it back to an uninitialized state.
    ///
    /// Has no effect and returns `None` if the `SyncOnceCell` hasn't been initialized.
    ///
    /// Safety is guaranteed by requiring a mutable reference.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(once_cell)]
    ///
    /// use std::lazy::SyncOnceCell;
    ///
    /// let mut cell: SyncOnceCell<String> = SyncOnceCell::new();
    /// assert_eq!(cell.take(), None);
    ///
    /// let mut cell = SyncOnceCell::new();
    /// cell.set("hello".to_string()).unwrap();
    /// assert_eq!(cell.take(), Some("hello".to_string()));
    /// assert_eq!(cell.get(), None);
    /// ```
    #[unstable(feature = "once_cell", issue = "74465")]
    pub fn take(&mut self) -> Option<T> {
        mem::take(self).into_inner()
    }

    /// Takes the wrapped value out of a `SyncOnceCell`.
    /// Afterwards the cell is no longer initialized.
    ///
    /// Safety: The cell must now be free'd WITHOUT dropping. No other usages of the cell
    /// are valid. Only used by `into_inner` and `drop`.
    unsafe fn take_inner(&mut self) -> Option<T> {
        // The mutable reference guarantees there are no other threads that can observe us
        // taking out the wrapped value.
        // Right after this function `self` is supposed to be freed, so it makes little sense
        // to atomically set the state to uninitialized.
        if self.is_initialized() {
            let value = mem::replace(&mut self.value, UnsafeCell::new(MaybeUninit::uninit()));
            Some(value.into_inner().assume_init())
        } else {
            None
        }
    }

    #[inline]
    fn is_initialized(&self) -> bool {
        self.once.is_completed()
    }

    #[cold]
    fn initialize<F, E>(&self, f: F) -> Result<(), E>
    where
        F: FnOnce() -> Result<T, E>,
    {
        let mut res: Result<(), E> = Ok(());
        let slot = &self.value;

        // Ignore poisoning from other threads
        // If another thread panics, then we'll be able to run our closure
        self.once.call_once_force(|p| {
            match f() {
                Ok(value) => {
                    unsafe { (&mut *slot.get()).write(value) };
                }
                Err(e) => {
                    res = Err(e);

                    // Treat the underlying `Once` as poisoned since we
                    // failed to initialize our value. Calls
                    p.poison();
                }
            }
        });
        res
    }

    /// Safety: The value must be initialized
    unsafe fn get_unchecked(&self) -> &T {
        debug_assert!(self.is_initialized());
        (&*self.value.get()).get_ref()
    }

    /// Safety: The value must be initialized
    unsafe fn get_unchecked_mut(&mut self) -> &mut T {
        debug_assert!(self.is_initialized());
        (&mut *self.value.get()).get_mut()
    }
}

unsafe impl<#[may_dangle] T> Drop for SyncOnceCell<T> {
    fn drop(&mut self) {
        // Safety: The cell is being dropped, so it can't be accessed again.
        // We also don't touch the `T`, which validates our usage of #[may_dangle].
        unsafe { self.take_inner() };
    }
}

/// A value which is initialized on the first access.
///
/// This type is a thread-safe `Lazy`, and can be used in statics.
///
/// # Examples
///
/// ```
/// #![feature(once_cell)]
///
/// use std::collections::HashMap;
///
/// use std::lazy::SyncLazy;
///
/// static HASHMAP: SyncLazy<HashMap<i32, String>> = SyncLazy::new(|| {
///     println!("initializing");
///     let mut m = HashMap::new();
///     m.insert(13, "Spica".to_string());
///     m.insert(74, "Hoyten".to_string());
///     m
/// });
///
/// fn main() {
///     println!("ready");
///     std::thread::spawn(|| {
///         println!("{:?}", HASHMAP.get(&13));
///     }).join().unwrap();
///     println!("{:?}", HASHMAP.get(&74));
///
///     // Prints:
///     //   ready
///     //   initializing
///     //   Some("Spica")
///     //   Some("Hoyten")
/// }
/// ```
#[unstable(feature = "once_cell", issue = "74465")]
pub struct SyncLazy<T, F = fn() -> T> {
    cell: SyncOnceCell<T>,
    init: Cell<Option<F>>,
}

#[unstable(feature = "once_cell", issue = "74465")]
impl<T: fmt::Debug, F> fmt::Debug for SyncLazy<T, F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Lazy").field("cell", &self.cell).field("init", &"..").finish()
    }
}

// We never create a `&F` from a `&SyncLazy<T, F>` so it is fine
// to not impl `Sync` for `F`
// we do create a `&mut Option<F>` in `force`, but this is
// properly synchronized, so it only happens once
// so it also does not contribute to this impl.
#[unstable(feature = "once_cell", issue = "74465")]
unsafe impl<T, F: Send> Sync for SyncLazy<T, F> where SyncOnceCell<T>: Sync {}
// auto-derived `Send` impl is OK.

#[unstable(feature = "once_cell", issue = "74465")]
impl<T, F: UnwindSafe> RefUnwindSafe for SyncLazy<T, F> where SyncOnceCell<T>: RefUnwindSafe {}
#[unstable(feature = "once_cell", issue = "74465")]
impl<T, F: UnwindSafe> UnwindSafe for SyncLazy<T, F> where SyncOnceCell<T>: UnwindSafe {}

impl<T, F> SyncLazy<T, F> {
    /// Creates a new lazy value with the given initializing
    /// function.
    #[unstable(feature = "once_cell", issue = "74465")]
    pub const fn new(f: F) -> SyncLazy<T, F> {
        SyncLazy { cell: SyncOnceCell::new(), init: Cell::new(Some(f)) }
    }
}

impl<T, F: FnOnce() -> T> SyncLazy<T, F> {
    /// Forces the evaluation of this lazy value and
    /// returns a reference to result. This is equivalent
    /// to the `Deref` impl, but is explicit.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(once_cell)]
    ///
    /// use std::lazy::SyncLazy;
    ///
    /// let lazy = SyncLazy::new(|| 92);
    ///
    /// assert_eq!(SyncLazy::force(&lazy), &92);
    /// assert_eq!(&*lazy, &92);
    /// ```
    #[unstable(feature = "once_cell", issue = "74465")]
    pub fn force(this: &SyncLazy<T, F>) -> &T {
        this.cell.get_or_init(|| match this.init.take() {
            Some(f) => f(),
            None => panic!("Lazy instance has previously been poisoned"),
        })
    }
}

#[unstable(feature = "once_cell", issue = "74465")]
impl<T, F: FnOnce() -> T> Deref for SyncLazy<T, F> {
    type Target = T;
    fn deref(&self) -> &T {
        SyncLazy::force(self)
    }
}

#[unstable(feature = "once_cell", issue = "74465")]
impl<T: Default> Default for SyncLazy<T> {
    /// Creates a new lazy value using `Default` as the initializing function.
    fn default() -> SyncLazy<T> {
        SyncLazy::new(T::default)
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        lazy::{Lazy, SyncLazy, SyncOnceCell},
        panic,
        sync::{
            atomic::{AtomicUsize, Ordering::SeqCst},
            mpsc::channel,
            Mutex,
        },
        thread,
    };

    #[test]
    fn lazy_default() {
        static CALLED: AtomicUsize = AtomicUsize::new(0);

        struct Foo(u8);
        impl Default for Foo {
            fn default() -> Self {
                CALLED.fetch_add(1, SeqCst);
                Foo(42)
            }
        }

        let lazy: Lazy<Mutex<Foo>> = <_>::default();

        assert_eq!(CALLED.load(SeqCst), 0);

        assert_eq!(lazy.lock().unwrap().0, 42);
        assert_eq!(CALLED.load(SeqCst), 1);

        lazy.lock().unwrap().0 = 21;

        assert_eq!(lazy.lock().unwrap().0, 21);
        assert_eq!(CALLED.load(SeqCst), 1);
    }

    #[test]
    fn lazy_poisoning() {
        let x: Lazy<String> = Lazy::new(|| panic!("kaboom"));
        for _ in 0..2 {
            let res = panic::catch_unwind(panic::AssertUnwindSafe(|| x.len()));
            assert!(res.is_err());
        }
    }

    fn spawn_and_wait<R: Send + 'static>(f: impl FnOnce() -> R + Send + 'static) -> R {
        thread::spawn(f).join().unwrap()
    }

    #[test]
    fn sync_once_cell() {
        static ONCE_CELL: SyncOnceCell<i32> = SyncOnceCell::new();

        assert!(ONCE_CELL.get().is_none());

        spawn_and_wait(|| {
            ONCE_CELL.get_or_init(|| 92);
            assert_eq!(ONCE_CELL.get(), Some(&92));
        });

        ONCE_CELL.get_or_init(|| panic!("Kabom!"));
        assert_eq!(ONCE_CELL.get(), Some(&92));
    }

    #[test]
    fn sync_once_cell_get_mut() {
        let mut c = SyncOnceCell::new();
        assert!(c.get_mut().is_none());
        c.set(90).unwrap();
        *c.get_mut().unwrap() += 2;
        assert_eq!(c.get_mut(), Some(&mut 92));
    }

    #[test]
    fn sync_once_cell_get_unchecked() {
        let c = SyncOnceCell::new();
        c.set(92).unwrap();
        unsafe {
            assert_eq!(c.get_unchecked(), &92);
        }
    }

    #[test]
    fn sync_once_cell_drop() {
        static DROP_CNT: AtomicUsize = AtomicUsize::new(0);
        struct Dropper;
        impl Drop for Dropper {
            fn drop(&mut self) {
                DROP_CNT.fetch_add(1, SeqCst);
            }
        }

        let x = SyncOnceCell::new();
        spawn_and_wait(move || {
            x.get_or_init(|| Dropper);
            assert_eq!(DROP_CNT.load(SeqCst), 0);
            drop(x);
        });

        assert_eq!(DROP_CNT.load(SeqCst), 1);
    }

    #[test]
    fn sync_once_cell_drop_empty() {
        let x = SyncOnceCell::<String>::new();
        drop(x);
    }

    #[test]
    fn clone() {
        let s = SyncOnceCell::new();
        let c = s.clone();
        assert!(c.get().is_none());

        s.set("hello".to_string()).unwrap();
        let c = s.clone();
        assert_eq!(c.get().map(String::as_str), Some("hello"));
    }

    #[test]
    fn get_or_try_init() {
        let cell: SyncOnceCell<String> = SyncOnceCell::new();
        assert!(cell.get().is_none());

        let res = panic::catch_unwind(|| cell.get_or_try_init(|| -> Result<_, ()> { panic!() }));
        assert!(res.is_err());
        assert!(!cell.is_initialized());
        assert!(cell.get().is_none());

        assert_eq!(cell.get_or_try_init(|| Err(())), Err(()));

        assert_eq!(
            cell.get_or_try_init(|| Ok::<_, ()>("hello".to_string())),
            Ok(&"hello".to_string())
        );
        assert_eq!(cell.get(), Some(&"hello".to_string()));
    }

    #[test]
    fn from_impl() {
        assert_eq!(SyncOnceCell::from("value").get(), Some(&"value"));
        assert_ne!(SyncOnceCell::from("foo").get(), Some(&"bar"));
    }

    #[test]
    fn partialeq_impl() {
        assert!(SyncOnceCell::from("value") == SyncOnceCell::from("value"));
        assert!(SyncOnceCell::from("foo") != SyncOnceCell::from("bar"));

        assert!(SyncOnceCell::<String>::new() == SyncOnceCell::new());
        assert!(SyncOnceCell::<String>::new() != SyncOnceCell::from("value".to_owned()));
    }

    #[test]
    fn into_inner() {
        let cell: SyncOnceCell<String> = SyncOnceCell::new();
        assert_eq!(cell.into_inner(), None);
        let cell = SyncOnceCell::new();
        cell.set("hello".to_string()).unwrap();
        assert_eq!(cell.into_inner(), Some("hello".to_string()));
    }

    #[test]
    fn sync_lazy_new() {
        static CALLED: AtomicUsize = AtomicUsize::new(0);
        static SYNC_LAZY: SyncLazy<i32> = SyncLazy::new(|| {
            CALLED.fetch_add(1, SeqCst);
            92
        });

        assert_eq!(CALLED.load(SeqCst), 0);

        spawn_and_wait(|| {
            let y = *SYNC_LAZY - 30;
            assert_eq!(y, 62);
            assert_eq!(CALLED.load(SeqCst), 1);
        });

        let y = *SYNC_LAZY - 30;
        assert_eq!(y, 62);
        assert_eq!(CALLED.load(SeqCst), 1);
    }

    #[test]
    fn sync_lazy_default() {
        static CALLED: AtomicUsize = AtomicUsize::new(0);

        struct Foo(u8);
        impl Default for Foo {
            fn default() -> Self {
                CALLED.fetch_add(1, SeqCst);
                Foo(42)
            }
        }

        let lazy: SyncLazy<Mutex<Foo>> = <_>::default();

        assert_eq!(CALLED.load(SeqCst), 0);

        assert_eq!(lazy.lock().unwrap().0, 42);
        assert_eq!(CALLED.load(SeqCst), 1);

        lazy.lock().unwrap().0 = 21;

        assert_eq!(lazy.lock().unwrap().0, 21);
        assert_eq!(CALLED.load(SeqCst), 1);
    }

    #[test]
    fn static_sync_lazy() {
        static XS: SyncLazy<Vec<i32>> = SyncLazy::new(|| {
            let mut xs = Vec::new();
            xs.push(1);
            xs.push(2);
            xs.push(3);
            xs
        });

        spawn_and_wait(|| {
            assert_eq!(&*XS, &vec![1, 2, 3]);
        });

        assert_eq!(&*XS, &vec![1, 2, 3]);
    }

    #[test]
    fn static_sync_lazy_via_fn() {
        fn xs() -> &'static Vec<i32> {
            static XS: SyncOnceCell<Vec<i32>> = SyncOnceCell::new();
            XS.get_or_init(|| {
                let mut xs = Vec::new();
                xs.push(1);
                xs.push(2);
                xs.push(3);
                xs
            })
        }
        assert_eq!(xs(), &vec![1, 2, 3]);
    }

    #[test]
    fn sync_lazy_poisoning() {
        let x: SyncLazy<String> = SyncLazy::new(|| panic!("kaboom"));
        for _ in 0..2 {
            let res = panic::catch_unwind(|| x.len());
            assert!(res.is_err());
        }
    }

    #[test]
    fn is_sync_send() {
        fn assert_traits<T: Send + Sync>() {}
        assert_traits::<SyncOnceCell<String>>();
        assert_traits::<SyncLazy<String>>();
    }

    #[test]
    fn eval_once_macro() {
        macro_rules! eval_once {
            (|| -> $ty:ty {
                $($body:tt)*
            }) => {{
                static ONCE_CELL: SyncOnceCell<$ty> = SyncOnceCell::new();
                fn init() -> $ty {
                    $($body)*
                }
                ONCE_CELL.get_or_init(init)
            }};
        }

        let fib: &'static Vec<i32> = eval_once! {
            || -> Vec<i32> {
                let mut res = vec![1, 1];
                for i in 0..10 {
                    let next = res[i] + res[i + 1];
                    res.push(next);
                }
                res
            }
        };
        assert_eq!(fib[5], 8)
    }

    #[test]
    fn sync_once_cell_does_not_leak_partially_constructed_boxes() {
        static ONCE_CELL: SyncOnceCell<String> = SyncOnceCell::new();

        let n_readers = 10;
        let n_writers = 3;
        const MSG: &str = "Hello, World";

        let (tx, rx) = channel();

        for _ in 0..n_readers {
            let tx = tx.clone();
            thread::spawn(move || {
                loop {
                    if let Some(msg) = ONCE_CELL.get() {
                        tx.send(msg).unwrap();
                        break;
                    }
                    #[cfg(target_env = "sgx")]
                    crate::thread::yield_now();
                }
            });
        }
        for _ in 0..n_writers {
            thread::spawn(move || {
                let _ = ONCE_CELL.set(MSG.to_owned());
            });
        }

        for _ in 0..n_readers {
            let msg = rx.recv().unwrap();
            assert_eq!(msg, MSG);
        }
    }

    #[test]
    fn dropck() {
        let cell = SyncOnceCell::new();
        {
            let s = String::new();
            cell.set(&s).unwrap();
        }
    }
}
