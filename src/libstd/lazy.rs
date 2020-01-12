//! `lazy` modules provides lazy values and one-time initialization of static data.
//!
//! `lazy` provides two new cell-like types, `OnceCell` and `SyncOnceCell`. `OnceCell`
//! might store arbitrary non-`Copy` types, can be assigned to at most once and provide direct access
//! to the stored contents. In a nutshell, API looks *roughly* like this:
//!
//! ```rust,ignore
//! impl<T> OnceCell<T> {
//!     fn new() -> OnceCell<T> { ... }
//!     fn set(&self, value: T) -> Result<(), T> { ... }
//!     fn get(&self) -> Option<&T> { ... }
//! }
//! ```
//!
//! Note that, like with `RefCell` and `Mutex`, the `set` method requires only a shared reference.
//! Because of the single assignment restriction `get` can return an `&T` instead of `Ref<T>`
//! or `MutexGuard<T>`.
//!
//! The `SyncOnceCell` flavor is thread-safe (that is, implements [`Sync`]) trait, while  `OnceCell` one is not.
//!
//! [`Sync`]: https://doc.rust-lang.org/std/marker/trait.Sync.html
//!
//! # Patterns
//!
//! `OnceCell` might be useful for a variety of patterns.
//!
//! ## Safe Initialization of global data
//!
//! ```rust
//! use std::{env, io};
//! use std::lazy::SyncOnceCell;
//!
//! #[derive(Debug)]
//! pub struct Logger {
//!     // ...
//! }
//! static INSTANCE: OnceCell<Logger> = OnceCell::new();
//!
//! impl Logger {
//!     pub fn global() -> &'static Logger {
//!         INSTANCE.get().expect("logger is not initialized")
//!     }
//!
//!     fn from_cli(args: env::Args) -> Result<Logger, std::io::Error> {
//!        // ...
//! #      Ok(Logger {})
//!     }
//! }
//!
//! fn main() {
//!     let logger = Logger::from_cli(env::args()).unwrap();
//!     INSTANCE.set(logger).unwrap();
//!     // use `Logger::global()` from now on
//! }
//! ```
//!
//! ## Lazy initialized global data
//!
//! This is essentially `lazy_static!` macro, but without a macro.
//!
//! ```rust
//! use std::{sync::Mutex, collections::HashMap};
//! use lazy::SyncOnceCell;
//!
//! fn global_data() -> &'static Mutex<HashMap<i32, String>> {
//!     static INSTANCE: OnceCell<Mutex<HashMap<i32, String>>> = OnceCell::new();
//!     INSTANCE.get_or_init(|| {
//!         let mut m = HashMap::new();
//!         m.insert(13, "Spica".to_string());
//!         m.insert(74, "Hoyten".to_string());
//!         Mutex::new(m)
//!     })
//! }
//! ```
//!
//! There are also `sync::Lazy` and `unsync::Lazy` convenience types to streamline this pattern:
//!
//! ```rust
//! use std::{sync::Mutex, collections::HashMap};
//! use lazy::SyncLazy;
//!
//! static GLOBAL_DATA: Lazy<Mutex<HashMap<i32, String>>> = Lazy::new(|| {
//!     let mut m = HashMap::new();
//!     m.insert(13, "Spica".to_string());
//!     m.insert(74, "Hoyten".to_string());
//!     Mutex::new(m)
//! });
//!
//! fn main() {
//!     println!("{:?}", GLOBAL_DATA.lock().unwrap());
//! }
//! ```
//!
//! ## General purpose lazy evaluation
//!
//! `Lazy` also works with local variables.
//!
//! ```rust
//! use std::lazy::Lazy;
//!
//! fn main() {
//!     let ctx = vec![1, 2, 3];
//!     let thunk = Lazy::new(|| {
//!         ctx.iter().sum::<i32>()
//!     });
//!     assert_eq!(*thunk, 6);
//! }
//! ```
//!
//! If you need a lazy field in a struct, you probably should use `OnceCell`
//! directly, because that will allow you to access `self` during initialization.
//!
//! ```rust
//! use std::{fs, path::PathBuf};
//!
//! use std::lazy::OnceCell;
//!
//! struct Ctx {
//!     config_path: PathBuf,
//!     config: OnceCell<String>,
//! }
//!
//! impl Ctx {
//!     pub fn get_config(&self) -> Result<&str, std::io::Error> {
//!         let cfg = self.config.get_or_try_init(|| {
//!             fs::read_to_string(&self.config_path)
//!         })?;
//!         Ok(cfg.as_str())
//!     }
//! }
//! ```
//!
//! ## Building block
//!
//! Naturally, it is  possible to build other abstractions on top of `OnceCell`.
//! For example, this is a `regex!` macro which takes a string literal and returns an
//! *expression* that evaluates to a `&'static Regex`:
//!
//! ```
//! macro_rules! regex {
//!     ($re:literal $(,)?) => {{
//!         static RE: std::lazy::SyncOnceCell<regex::Regex> = std::lazy::SyncOnceCell::new();
//!         RE.get_or_init(|| regex::Regex::new($re).unwrap())
//!     }};
//! }
//! ```
//!
//! This macro can be useful to avoid "compile regex on every loop iteration" problem.
//!
//! # Comparison with other interior mutatbility types
//!
//! |`!Sync` types         | Access Mode            | Drawbacks                                     |
//! |----------------------|------------------------|-----------------------------------------------|
//! |`Cell<T>`             | `T`                    | requires `T: Copy` for `get`                  |
//! |`RefCell<T>`          | `RefMut<T>` / `Ref<T>` | may panic at runtime                          |
//! |`OnceCell<T>`         | `&T`                   | assignable only once                          |
//!
//! |`Sync` types          | Access Mode            | Drawbacks                                     |
//! |----------------------|------------------------|-----------------------------------------------|
//! |`AtomicT`             | `T`                    | works only with certain `Copy` types          |
//! |`Mutex<T>`            | `MutexGuard<T>`        | may deadlock at runtime, may block the thread |
//! |`SyncOnceCell<T>`     | `&T`                   | assignable only once, may block the thread    |
//!
//! Technically, calling `get_or_init` will also cause a panic or a deadlock if it recursively calls
//! itself. However, because the assignment can happen only once, such cases should be more rare than
//! equivalents with `RefCell` and `Mutex`.

use crate::{
    cell::{Cell, UnsafeCell},
    fmt,
    hint::unreachable_unchecked,
    marker::PhantomData,
    ops::Deref,
    panic::{RefUnwindSafe, UnwindSafe},
    sync::atomic::{AtomicBool, AtomicUsize, Ordering},
    thread::{self, Thread},
};

/// A cell which can be written to only once. Not thread safe.
///
/// Unlike `:td::cell::RefCell`, a `OnceCell` provides simple `&`
/// references to the contents.
///
/// # Example
/// ```
/// use std::lazy::OnceCell;
///
/// let cell = OnceCell::new();
/// assert!(cell.get().is_none());
///
/// let value: &String = cell.get_or_init(|| {
///     "Hello, World!".to_string()
/// });
/// assert_eq!(value, "Hello, World!");
/// assert!(cell.get().is_some());
/// ```
pub struct OnceCell<T> {
    // Invariant: written to at most once.
    inner: UnsafeCell<Option<T>>,
}

// Similarly to a `Sync` bound on `SyncOnceCell`, we can use
// `&OnceCell` to sneak a `T` through `catch_unwind`,
// by initializing the cell in closure and extracting the value in the
// `Drop`.
#[cfg(feature = "std")]
impl<T: RefUnwindSafe + UnwindSafe> RefUnwindSafe for OnceCell<T> {}
#[cfg(feature = "std")]
impl<T: UnwindSafe> UnwindSafe for OnceCell<T> {}

impl<T> Default for OnceCell<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: fmt::Debug> fmt::Debug for OnceCell<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.get() {
            Some(v) => f.debug_tuple("OnceCell").field(v).finish(),
            None => f.write_str("OnceCell(Uninit)"),
        }
    }
}

impl<T: Clone> Clone for OnceCell<T> {
    fn clone(&self) -> OnceCell<T> {
        let res = OnceCell::new();
        if let Some(value) = self.get() {
            match res.set(value.clone()) {
                Ok(()) => (),
                Err(_) => unreachable!(),
            }
        }
        res
    }
}

impl<T: PartialEq> PartialEq for OnceCell<T> {
    fn eq(&self, other: &Self) -> bool {
        self.get() == other.get()
    }
}

impl<T: Eq> Eq for OnceCell<T> {}

impl<T> From<T> for OnceCell<T> {
    fn from(value: T) -> Self {
        OnceCell { inner: UnsafeCell::new(Some(value)) }
    }
}

impl<T> OnceCell<T> {
    /// Creates a new empty cell.
    pub const fn new() -> OnceCell<T> {
        OnceCell { inner: UnsafeCell::new(None) }
    }

    /// Gets the reference to the underlying value.
    ///
    /// Returns `None` if the cell is empty.
    pub fn get(&self) -> Option<&T> {
        // Safe due to `inner`'s invariant
        unsafe { &*self.inner.get() }.as_ref()
    }

    /// Gets the mutable reference to the underlying value.
    ///
    /// Returns `None` if the cell is empty.
    pub fn get_mut(&mut self) -> Option<&mut T> {
        // Safe because we have unique access
        unsafe { &mut *self.inner.get() }.as_mut()
    }

    /// Sets the contents of this cell to `value`.
    ///
    /// Returns `Ok(())` if the cell was empty and `Err(value)` if it was
    /// full.
    ///
    /// # Example
    /// ```
    /// use std::lazy::OnceCell;
    ///
    /// let cell = OnceCell::new();
    /// assert!(cell.get().is_none());
    ///
    /// assert_eq!(cell.set(92), Ok(()));
    /// assert_eq!(cell.set(62), Err(62));
    ///
    /// assert!(cell.get().is_some());
    /// ```
    pub fn set(&self, value: T) -> Result<(), T> {
        let slot = unsafe { &*self.inner.get() };
        if slot.is_some() {
            return Err(value);
        }
        let slot = unsafe { &mut *self.inner.get() };
        // This is the only place where we set the slot, no races
        // due to reentrancy/concurrency are possible, and we've
        // checked that slot is currently `None`, so this write
        // maintains the `inner`'s invariant.
        *slot = Some(value);
        Ok(())
    }

    /// Gets the contents of the cell, initializing it with `f`
    /// if the cell was empty.
    ///
    /// # Panics
    ///
    /// If `f` panics, the panic is propagated to the caller, and the cell
    /// remains uninitialized.
    ///
    /// It is an error to reentrantly initialize the cell from `f`. Doing
    /// so results in a panic.
    ///
    /// # Example
    /// ```
    /// use std::lazy::OnceCell;
    ///
    /// let cell = OnceCell::new();
    /// let value = cell.get_or_init(|| 92);
    /// assert_eq!(value, &92);
    /// let value = cell.get_or_init(|| unreachable!());
    /// assert_eq!(value, &92);
    /// ```
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
    /// If `f` panics, the panic is propagated to the caller, and the cell
    /// remains uninitialized.
    ///
    /// It is an error to reentrantly initialize the cell from `f`. Doing
    /// so results in a panic.
    ///
    /// # Example
    /// ```
    /// use std::lazy::OnceCell;
    ///
    /// let cell = OnceCell::new();
    /// assert_eq!(cell.get_or_try_init(|| Err(())), Err(()));
    /// assert!(cell.get().is_none());
    /// let value = cell.get_or_try_init(|| -> Result<i32, ()> {
    ///     Ok(92)
    /// });
    /// assert_eq!(value, Ok(&92));
    /// assert_eq!(cell.get(), Some(&92))
    /// ```
    pub fn get_or_try_init<F, E>(&self, f: F) -> Result<&T, E>
    where
        F: FnOnce() -> Result<T, E>,
    {
        if let Some(val) = self.get() {
            return Ok(val);
        }
        let val = f()?;
        // Note that *some* forms of reentrant initialization might lead to
        // UB (see `reentrant_init` test). I believe that just removing this
        // `assert`, while keeping `set/get` would be sound, but it seems
        // better to panic, rather than to silently use an old value.
        assert!(self.set(val).is_ok(), "reentrant init");
        Ok(self.get().unwrap())
    }

    /// Consumes the `OnceCell`, returning the wrapped value.
    ///
    /// Returns `None` if the cell was empty.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::lazy::OnceCell;
    ///
    /// let cell: OnceCell<String> = OnceCell::new();
    /// assert_eq!(cell.into_inner(), None);
    ///
    /// let cell = OnceCell::new();
    /// cell.set("hello".to_string()).unwrap();
    /// assert_eq!(cell.into_inner(), Some("hello".to_string()));
    /// ```
    pub fn into_inner(self) -> Option<T> {
        // Because `into_inner` takes `self` by value, the compiler statically verifies
        // that it is not currently borrowed. So it is safe to move out `Option<T>`.
        self.inner.into_inner()
    }
}

/// A value which is initialized on the first access.
///
/// # Example
/// ```
/// use std::lazy::Lazy;
///
/// let lazy: Lazy<i32> = Lazy::new(|| {
///     println!("initializing");
///     92
/// });
/// println!("ready");
/// println!("{}", *lazy);
/// println!("{}", *lazy);
///
/// // Prints:
/// //   ready
/// //   initializing
/// //   92
/// //   92
/// ```
pub struct Lazy<T, F = fn() -> T> {
    cell: OnceCell<T>,
    init: Cell<Option<F>>,
}

#[cfg(feature = "std")]
impl<T, F: RefUnwindSafe> RefUnwindSafe for Lazy<T, F> where OnceCell<T>: RefUnwindSafe {}

impl<T: fmt::Debug, F: fmt::Debug> fmt::Debug for Lazy<T, F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Lazy").field("cell", &self.cell).field("init", &"..").finish()
    }
}

impl<T, F> Lazy<T, F> {
    /// Creates a new lazy value with the given initializing function.
    ///
    /// # Example
    /// ```
    /// # fn main() {
    /// use std::lazy::Lazy;
    ///
    /// let hello = "Hello, World!".to_string();
    ///
    /// let lazy = Lazy::new(|| hello.to_uppercase());
    ///
    /// assert_eq!(&*lazy, "HELLO, WORLD!");
    /// # }
    /// ```
    pub const fn new(init: F) -> Lazy<T, F> {
        Lazy { cell: OnceCell::new(), init: Cell::new(Some(init)) }
    }
}

impl<T, F: FnOnce() -> T> Lazy<T, F> {
    /// Forces the evaluation of this lazy value and returns a reference to
    /// the result.
    ///
    /// This is equivalent to the `Deref` impl, but is explicit.
    ///
    /// # Example
    /// ```
    /// use std::lazy::Lazy;
    ///
    /// let lazy = Lazy::new(|| 92);
    ///
    /// assert_eq!(Lazy::force(&lazy), &92);
    /// assert_eq!(&*lazy, &92);
    /// ```
    pub fn force(this: &Lazy<T, F>) -> &T {
        this.cell.get_or_init(|| match this.init.take() {
            Some(f) => f(),
            None => panic!("Lazy instance has previously been poisoned"),
        })
    }
}

impl<T, F: FnOnce() -> T> Deref for Lazy<T, F> {
    type Target = T;
    fn deref(&self) -> &T {
        Lazy::force(self)
    }
}

impl<T: Default> Default for Lazy<T> {
    /// Creates a new lazy value using `Default` as the initializing function.
    fn default() -> Lazy<T> {
        Lazy::new(T::default)
    }
}

/// A thread-safe cell which can be written to only once.
///
/// `OnceCell` provides `&` references to the contents without RAII guards.
///
/// Reading a non-`None` value out of `OnceCell` establishes a
/// happens-before relationship with a corresponding write. For example, if
/// thread A initializes the cell with `get_or_init(f)`, and thread B
/// subsequently reads the result of this call, B also observes all the side
/// effects of `f`.
///
/// # Example
/// ```
/// use std::lazy::SyncOnceCell;
///
/// static CELL: OnceCell<String> = OnceCell::new();
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
pub struct SyncOnceCell<T> {
    // This `state` word is actually an encoded version of just a pointer to a
    // `Waiter`, so we add the `PhantomData` appropriately.
    state_and_queue: AtomicUsize,
    _marker: PhantomData<*mut Waiter>,
    // FIXME: switch to `std::mem::MaybeUninit` once we are ready to bump MSRV
    // that far. It was stabilized in 1.36.0, so, if you are reading this and
    // it's higher than 1.46.0 outside, please send a PR! ;) (and do the same
    // for `Lazy`, while we are at it).
    pub(crate) value: UnsafeCell<Option<T>>,
}

// Why do we need `T: Send`?
// Thread A creates a `OnceCell` and shares it with
// scoped thread B, which fills the cell, which is
// then destroyed by A. That is, destructor observes
// a sent value.
unsafe impl<T: Sync + Send> Sync for SyncOnceCell<T> {}
unsafe impl<T: Send> Send for SyncOnceCell<T> {}

impl<T: RefUnwindSafe + UnwindSafe> RefUnwindSafe for SyncOnceCell<T> {}
impl<T: UnwindSafe> UnwindSafe for SyncOnceCell<T> {}

impl<T> Default for SyncOnceCell<T> {
    fn default() -> SyncOnceCell<T> {
        SyncOnceCell::new()
    }
}

impl<T: fmt::Debug> fmt::Debug for SyncOnceCell<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.get() {
            Some(v) => f.debug_tuple("SyncOnceCell").field(v).finish(),
            None => f.write_str("SyncOnceCell(Uninit)"),
        }
    }
}

impl<T: Clone> Clone for SyncOnceCell<T> {
    fn clone(&self) -> SyncOnceCell<T> {
        let res = SyncOnceCell::new();
        if let Some(value) = self.get() {
            match res.set(value.clone()) {
                Ok(()) => (),
                Err(_) => unreachable!(),
            }
        }
        res
    }
}

impl<T> From<T> for SyncOnceCell<T> {
    fn from(value: T) -> Self {
        let cell = Self::new();
        cell.get_or_init(|| value);
        cell
    }
}

impl<T: PartialEq> PartialEq for SyncOnceCell<T> {
    fn eq(&self, other: &SyncOnceCell<T>) -> bool {
        self.get() == other.get()
    }
}

impl<T: Eq> Eq for SyncOnceCell<T> {}

impl<T> SyncOnceCell<T> {
    /// Creates a new empty cell.
    pub const fn new() -> SyncOnceCell<T> {
        SyncOnceCell {
            state_and_queue: AtomicUsize::new(INCOMPLETE),
            _marker: PhantomData,
            value: UnsafeCell::new(None),
        }
    }

    /// Gets the reference to the underlying value.
    ///
    /// Returns `None` if the cell is empty, or being initialized. This
    /// method never blocks.
    pub fn get(&self) -> Option<&T> {
        if self.is_initialized() {
            // Safe b/c checked is_initialize
            Some(unsafe { self.get_unchecked() })
        } else {
            None
        }
    }

    /// Gets the mutable reference to the underlying value.
    ///
    /// Returns `None` if the cell is empty.
    pub fn get_mut(&mut self) -> Option<&mut T> {
        // Safe b/c we have a unique access.
        unsafe { &mut *self.value.get() }.as_mut()
    }

    /// Get the reference to the underlying value, without checking if the
    /// cell is initialized.
    ///
    /// Safety:
    ///
    /// Caller must ensure that the cell is in initialized state, and that
    /// the contents are acquired by (synchronized to) this thread.
    pub unsafe fn get_unchecked(&self) -> &T {
        debug_assert!(self.is_initialized());
        let slot: &Option<T> = &*self.value.get();
        match slot {
            Some(value) => value,
            // This unsafe does improve performance, see `examples/bench`.
            None => {
                debug_assert!(false);
                unreachable_unchecked()
            }
        }
    }

    /// Sets the contents of this cell to `value`.
    ///
    /// Returns `Ok(())` if the cell was empty and `Err(value)` if it was
    /// full.
    ///
    /// # Example
    /// ```
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
    /// # Example
    /// ```
    /// use std::lazy::SyncOnceCell;
    ///
    /// let cell = SyncOnceCell::new();
    /// let value = cell.get_or_init(|| 92);
    /// assert_eq!(value, &92);
    /// let value = cell.get_or_init(|| unreachable!());
    /// assert_eq!(value, &92);
    /// ```
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
    /// # Example
    /// ```
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
    pub fn get_or_try_init<F, E>(&self, f: F) -> Result<&T, E>
    where
        F: FnOnce() -> Result<T, E>,
    {
        // Fast path check
        if let Some(value) = self.get() {
            return Ok(value);
        }
        self.initialize(f)?;

        // Safe b/c called initialize
        debug_assert!(self.is_initialized());
        Ok(unsafe { self.get_unchecked() })
    }

    /// Consumes the `SyncOnceCell`, returning the wrapped value. Returns
    /// `None` if the cell was empty.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::lazy::SyncOnceCell;
    ///
    /// let cell: SyncOnceCell<String> = SyncOnceCell::new();
    /// assert_eq!(cell.into_inner(), None);
    ///
    /// let cell = SyncOnceCell::new();
    /// cell.set("hello".to_string()).unwrap();
    /// assert_eq!(cell.into_inner(), Some("hello".to_string()));
    /// ```
    pub fn into_inner(self) -> Option<T> {
        // Because `into_inner` takes `self` by value, the compiler statically verifies
        // that it is not currently borrowed. So it is safe to move out `Option<T>`.
        self.value.into_inner()
    }

    /// Safety: synchronizes with store to value via Release/(Acquire|SeqCst).
    #[inline]
    fn is_initialized(&self) -> bool {
        // An `Acquire` load is enough because that makes all the initialization
        // operations visible to us, and, this being a fast path, weaker
        // ordering helps with performance. This `Acquire` synchronizes with
        // `SeqCst` operations on the slow path.
        self.state_and_queue.load(Ordering::Acquire) == COMPLETE
    }

    /// Safety: synchronizes with store to value via SeqCst read from state,
    /// writes value only once because we never get to INCOMPLETE state after a
    /// successful write.
    #[cold]
    fn initialize<F, E>(&self, f: F) -> Result<(), E>
    where
        F: FnOnce() -> Result<T, E>,
    {
        let mut f = Some(f);
        let mut res: Result<(), E> = Ok(());
        let slot = &self.value;
        initialize_inner(&self.state_and_queue, &mut || {
            let f = f.take().unwrap();
            match f() {
                Ok(value) => {
                    unsafe { *slot.get() = Some(value) };
                    true
                }
                Err(e) => {
                    res = Err(e);
                    false
                }
            }
        });
        res
    }
}

// region: copy-paste
// The following code is copied from `sync::Once`.
// This should be uncopypasted once we decide the right way to handle panics.
const INCOMPLETE: usize = 0x0;
const RUNNING: usize = 0x1;
const COMPLETE: usize = 0x2;

const STATE_MASK: usize = 0x3;


#[repr(align(4))]
struct Waiter {
    thread: Cell<Option<Thread>>,
    signaled: AtomicBool,
    next: *const Waiter,
}

struct WaiterQueue<'a> {
    state_and_queue: &'a AtomicUsize,
    set_state_on_drop_to: usize,
}

impl Drop for WaiterQueue<'_> {
    fn drop(&mut self) {
        let state_and_queue =
            self.state_and_queue.swap(self.set_state_on_drop_to, Ordering::AcqRel);

        assert_eq!(state_and_queue & STATE_MASK, RUNNING);

        unsafe {
            let mut queue = (state_and_queue & !STATE_MASK) as *const Waiter;
            while !queue.is_null() {
                let next = (*queue).next;
                let thread = (*queue).thread.replace(None).unwrap();
                (*queue).signaled.store(true, Ordering::Release);
                queue = next;
                thread.unpark();
            }
        }
    }
}

fn initialize_inner(my_state_and_queue: &AtomicUsize, init: &mut dyn FnMut() -> bool) -> bool {
    let mut state_and_queue = my_state_and_queue.load(Ordering::Acquire);

    loop {
        match state_and_queue {
            COMPLETE => return true,
            INCOMPLETE => {
                let old = my_state_and_queue.compare_and_swap(
                    state_and_queue,
                    RUNNING,
                    Ordering::Acquire,
                );
                if old != state_and_queue {
                    state_and_queue = old;
                    continue;
                }
                let mut waiter_queue = WaiterQueue {
                    state_and_queue: my_state_and_queue,
                    set_state_on_drop_to: INCOMPLETE,
                };
                let success = init();

                waiter_queue.set_state_on_drop_to = if success { COMPLETE } else { INCOMPLETE };
                return success;
            }
            _ => {
                assert!(state_and_queue & STATE_MASK == RUNNING);
                wait(&my_state_and_queue, state_and_queue);
                state_and_queue = my_state_and_queue.load(Ordering::Acquire);
            }
        }
    }
}

fn wait(state_and_queue: &AtomicUsize, mut current_state: usize) {
    loop {
        if current_state & STATE_MASK != RUNNING {
            return;
        }

        let node = Waiter {
            thread: Cell::new(Some(thread::current())),
            signaled: AtomicBool::new(false),
            next: (current_state & !STATE_MASK) as *const Waiter,
        };
        let me = &node as *const Waiter as usize;

        let old = state_and_queue.compare_and_swap(current_state, me | RUNNING, Ordering::Release);
        if old != current_state {
            current_state = old;
            continue;
        }

        while !node.signaled.load(Ordering::Acquire) {
            thread::park();
        }
        break;
    }
}
// endregion: copy-paste

/// A value which is initialized on the first access.
///
/// This type is thread-safe and can be used in statics:
///
/// # Example
/// ```
/// use std::collections::HashMap;
///
/// use std::lazy::Lazy;
///
/// static HASHMAP: Lazy<HashMap<i32, String>> = Lazy::new(|| {
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
pub struct SyncLazy<T, F = fn() -> T> {
    cell: SyncOnceCell<T>,
    init: Cell<Option<F>>,
}

impl<T: fmt::Debug, F: fmt::Debug> fmt::Debug for SyncLazy<T, F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SyncLazy").field("cell", &self.cell).field("init", &"..").finish()
    }
}

// We never create a `&F` from a `&SyncLazy<T, F>` so it is fine
// to not impl `Sync` for `F`
// we do create a `&mut Option<F>` in `force`, but this is
// properly synchronized, so it only happens once
// so it also does not contribute to this impl.
unsafe impl<T, F: Send> Sync for SyncLazy<T, F> where SyncOnceCell<T>: Sync {}
// auto-derived `Send` impl is OK.

#[cfg(feature = "std")]
impl<T, F: RefUnwindSafe> RefUnwindSafe for SyncLazy<T, F> where SyncOnceCell<T>: RefUnwindSafe {}

impl<T, F> SyncLazy<T, F> {
    /// Creates a new lazy value with the given initializing
    /// function.
    pub const fn new(f: F) -> SyncLazy<T, F> {
        SyncLazy { cell: SyncOnceCell::new(), init: Cell::new(Some(f)) }
    }
}

impl<T, F: FnOnce() -> T> SyncLazy<T, F> {
    /// Forces the evaluation of this lazy value and
    /// returns a reference to result. This is equivalent
    /// to the `Deref` impl, but is explicit.
    ///
    /// # Example
    /// ```
    /// use std::lazy::SyncLazy;
    ///
    /// let lazy = SyncLazy::new(|| 92);
    ///
    /// assert_eq!(SyncLazy::force(&lazy), &92);
    /// assert_eq!(&*lazy, &92);
    /// ```
    pub fn force(this: &SyncLazy<T, F>) -> &T {
        this.cell.get_or_init(|| match this.init.take() {
            Some(f) => f(),
            None => panic!("SyncLazy instance has previously been poisoned"),
        })
    }
}

impl<T, F: FnOnce() -> T> Deref for SyncLazy<T, F> {
    type Target = T;
    fn deref(&self) -> &T {
        SyncLazy::force(self)
    }
}

impl<T: Default> Default for SyncLazy<T> {
    /// Creates a new lazy value using `Default` as the initializing function.
    fn default() -> SyncLazy<T> {
        SyncLazy::new(T::default)
    }
}
