//! Lazy values and one-time initialization of static data.
//!
//! `lazy` provides two new cell-like types, `Once` and `Lazy`. `Once`
//! might store arbitrary non-`Copy` types, can be assigned to at most once and provide direct access
//! to the stored contents.
//!
//! Note that, like with `RefCell` and `Mutex`, the `set` method requires only a shared reference.
//! Because of the single assignment restriction `get` can return an `&T` instead of `Ref<T>`
//! or `MutexGuard<T>`.
//!
//! # Patterns
//!
//! `Once` can be useful for a variety of patterns.
//!
//! ## Safe Initialization of global data
//!
//! ```rust
//! #![feature(once_cell)]
//!
//! use std::{env, io};
//! use std::lazy::Once;
//!
//! #[derive(Debug)]
//! pub struct Logger {
//!     // ...
//! }
//! static INSTANCE: Once<Logger> = Once::new();
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
//! #![feature(once_cell)]
//!
//! use std::{sync::Mutex, collections::HashMap};
//! use lazy::Once;
//!
//! fn global_data() -> &'static Mutex<HashMap<i32, String>> {
//!     static INSTANCE: Once<Mutex<HashMap<i32, String>>> = Once::new();
//!     INSTANCE.get_or_init(|| {
//!         let mut m = HashMap::new();
//!         m.insert(13, "Spica".to_string());
//!         m.insert(74, "Hoyten".to_string());
//!         Mutex::new(m)
//!     })
//! }
//! ```
//!
//! There is also `Lazy` to streamline this pattern:
//!
//! ```rust
//! #![feature(once_cell)]
//!
//! use std::{sync::Mutex, collections::HashMap};
//! use lazy::Lazy;
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
//! #![feature(once_cell)]
//!
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
//! If you need a lazy field in a struct, you probably should use `Once`
//! directly, because that will allow you to access `self` during initialization.
//!
//! ```rust
//! #![feature(once_cell)]
//!
//! use std::{fs, path::PathBuf};
//!
//! use std::lazy::Once;
//!
//! struct Ctx {
//!     config_path: PathBuf,
//!     config: Once<String>,
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
//! Naturally, it is  possible to build other abstractions on top of `Once`.
//! For example, this is a `regex!` macro which takes a string literal and returns an
//! *expression* that evaluates to a `&'static Regex`:
//!
//! ```
//! macro_rules! regex {
//!     ($re:literal $(,)?) => {{
//!         static RE: std::lazy::Once<regex::Regex> = std::lazy::Once::new();
//!         RE.get_or_init(|| regex::Regex::new($re).unwrap())
//!     }};
//! }
//! ```
//!
//! This macro can be useful to avoid "compile regex on every loop iteration" problem.
//!
//! # Comparison with other interior mutability types
//!
//! |`!Sync` types         | Access Mode            | Drawbacks                                     |
//! |----------------------|------------------------|-----------------------------------------------|
//! |`Cell<T>`             | `T`                    | requires `T: Copy` for `get`                  |
//! |`RefCell<T>`          | `RefMut<T>` / `Ref<T>` | may panic at runtime                          |
//! |`OnceCell<T>`         | `&T`                   | assignable only once                          |
//! |`LazyCell<T, F>`      | `&T`                   | assignable only once                          |
//!
//! |`Sync` types          | Access Mode            | Drawbacks                                     |
//! |----------------------|------------------------|-----------------------------------------------|
//! |`AtomicT`             | `T`                    | works only with certain `Copy` types          |
//! |`Mutex<T>`            | `MutexGuard<T>`        | may deadlock at runtime, may block the thread |
//! |`Once<T>`             | `&T`                   | assignable only once, may block the thread    |
//! |`Lazy<T, F>`          | `&T`                   | assignable only once, may block the thread    |
//!
//! Technically, calling `get_or_init` will also cause a panic or a deadlock if it recursively calls
//! itself. However, because the assignment can happen only once, such cases should be more rare than
//! equivalents with `RefCell` and `Mutex`.

use crate::{
    cell::{Cell, UnsafeCell},
    fmt,
    marker::PhantomData,
    mem::{self, MaybeUninit},
    ops::{Deref, Drop},
    panic::{RefUnwindSafe, UnwindSafe},
    sync::atomic::{AtomicBool, AtomicUsize, Ordering},
    thread::{self, Thread},
};

/// A synchronization primitive which can be written to only once.
///
/// This type is a thread-safe `OnceCell`.
///
/// # Examples
///
/// ```
/// #![feature(once_cell)]
///
/// use std::lazy::Once;
///
/// static CELL: Once<String> = Once::new();
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
#[unstable(feature = "once_cell", issue = "68198")]
pub struct Once<T> {
    // This `state` word is actually an encoded version of just a pointer to a
    // `Waiter`, so we add the `PhantomData` appropriately.
    state_and_queue: AtomicUsize,
    _marker: PhantomData<*mut Waiter>,
    // Whether or not the value is initialized is tracked by `state_and_queue`.
    value: UnsafeCell<MaybeUninit<T>>,
}

// Why do we need `T: Send`?
// Thread A creates a `Once` and shares it with
// scoped thread B, which fills the cell, which is
// then destroyed by A. That is, destructor observes
// a sent value.
#[unstable(feature = "once_cell", issue = "68198")]
unsafe impl<T: Sync + Send> Sync for Once<T> {}
#[unstable(feature = "once_cell", issue = "68198")]
unsafe impl<T: Send> Send for Once<T> {}

#[unstable(feature = "once_cell", issue = "68198")]
impl<T: RefUnwindSafe + UnwindSafe> RefUnwindSafe for Once<T> {}
#[unstable(feature = "once_cell", issue = "68198")]
impl<T: UnwindSafe> UnwindSafe for Once<T> {}

#[unstable(feature = "once_cell", issue = "68198")]
impl<T> Default for Once<T> {
    fn default() -> Once<T> {
        Once::new()
    }
}

#[unstable(feature = "once_cell", issue = "68198")]
impl<T: fmt::Debug> fmt::Debug for Once<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.get() {
            Some(v) => f.debug_tuple("Once").field(v).finish(),
            None => f.write_str("Once(Uninit)"),
        }
    }
}

#[unstable(feature = "once_cell", issue = "68198")]
impl<T: Clone> Clone for Once<T> {
    fn clone(&self) -> Once<T> {
        let res = Once::new();
        if let Some(value) = self.get() {
            match res.set(value.clone()) {
                Ok(()) => (),
                Err(_) => unreachable!(),
            }
        }
        res
    }
}

#[unstable(feature = "once_cell", issue = "68198")]
impl<T> From<T> for Once<T> {
    fn from(value: T) -> Self {
        let cell = Self::new();
        cell.get_or_init(|| value);
        cell
    }
}

#[unstable(feature = "once_cell", issue = "68198")]
impl<T: PartialEq> PartialEq for Once<T> {
    fn eq(&self, other: &Once<T>) -> bool {
        self.get() == other.get()
    }
}

#[unstable(feature = "once_cell", issue = "68198")]
impl<T: Eq> Eq for Once<T> {}

impl<T> Once<T> {
    /// Creates a new empty cell.
    #[unstable(feature = "once_cell", issue = "68198")]
    pub const fn new() -> Once<T> {
        Once {
            state_and_queue: AtomicUsize::new(INCOMPLETE),
            _marker: PhantomData,
            value: UnsafeCell::new(MaybeUninit::uninit()),
        }
    }

    /// Gets the reference to the underlying value.
    ///
    /// Returns `None` if the cell is empty, or being initialized. This
    /// method never blocks.
    #[unstable(feature = "once_cell", issue = "68198")]
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
    #[unstable(feature = "once_cell", issue = "68198")]
    pub fn get_mut(&mut self) -> Option<&mut T> {
        if self.is_initialized() {
            // Safe b/c checked is_initialize and we have a unique access
            Some(unsafe { self.get_unchecked_mut() })
        } else {
            None
        }
    }

    /// Sets the contents of this cell to `value`.
    ///
    /// Returns `Ok(())` if the cell was empty and `Err(value)` if it was
    /// full.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(once_cell)]
    ///
    /// use std::lazy::Once;
    ///
    /// static CELL: Once<i32> = Once::new();
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
    #[unstable(feature = "once_cell", issue = "68198")]
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
    /// use std::lazy::Once;
    ///
    /// let cell = Once::new();
    /// let value = cell.get_or_init(|| 92);
    /// assert_eq!(value, &92);
    /// let value = cell.get_or_init(|| unreachable!());
    /// assert_eq!(value, &92);
    /// ```
    #[unstable(feature = "once_cell", issue = "68198")]
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
    /// use std::lazy::Once;
    ///
    /// let cell = Once::new();
    /// assert_eq!(cell.get_or_try_init(|| Err(())), Err(()));
    /// assert!(cell.get().is_none());
    /// let value = cell.get_or_try_init(|| -> Result<i32, ()> {
    ///     Ok(92)
    /// });
    /// assert_eq!(value, Ok(&92));
    /// assert_eq!(cell.get(), Some(&92))
    /// ```
    #[unstable(feature = "once_cell", issue = "68198")]
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

    /// Consumes the `Once`, returning the wrapped value. Returns
    /// `None` if the cell was empty.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(once_cell)]
    ///
    /// use std::lazy::Once;
    ///
    /// let cell: Once<String> = Once::new();
    /// assert_eq!(cell.into_inner(), None);
    ///
    /// let cell = Once::new();
    /// cell.set("hello".to_string()).unwrap();
    /// assert_eq!(cell.into_inner(), Some("hello".to_string()));
    /// ```
    #[unstable(feature = "once_cell", issue = "68198")]
    pub fn into_inner(mut self) -> Option<T> {
        // Safety: Safe because we immediately free `self` without dropping
        let inner = unsafe { self.take_inner() };

        // Don't drop this `Once`. We just moved out one of the fields, but didn't set
        // the state to uninitialized.
        mem::ManuallyDrop::new(self);
        inner
    }

    /// Takes the wrapped value out of a `Once`.
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
                    unsafe { (&mut *slot.get()).write(value) };
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

impl<T> Drop for Once<T> {
    fn drop(&mut self) {
        // Safety: The cell is being dropped, so it can't be accessed again
        unsafe { self.take_inner() };
    }
}

// FIXME: The following code is copied from `sync::Once`.
// This should be uncopypasted once we decide the right way to handle panics.
// Do we want to effectively move the `Once` synchronization here and make `Once`
// a newtype: `pub struct Once(lazy::Once<()>)`?
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

/// A value which is initialized on the first access.
///
/// This type is a thread-safe `LazyCell`, and can be used in statics.
///
/// # Examples
///
/// ```
/// #![feature(once_cell)]
///
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
#[unstable(feature = "once_cell", issue = "68198")]
pub struct Lazy<T, F = fn() -> T> {
    cell: Once<T>,
    init: Cell<Option<F>>,
}

#[unstable(feature = "once_cell", issue = "68198")]
impl<T: fmt::Debug, F: fmt::Debug> fmt::Debug for Lazy<T, F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Lazy").field("cell", &self.cell).field("init", &"..").finish()
    }
}

// We never create a `&F` from a `&Lazy<T, F>` so it is fine
// to not impl `Sync` for `F`
// we do create a `&mut Option<F>` in `force`, but this is
// properly synchronized, so it only happens once
// so it also does not contribute to this impl.
#[unstable(feature = "once_cell", issue = "68198")]
unsafe impl<T, F: Send> Sync for Lazy<T, F> where Once<T>: Sync {}
// auto-derived `Send` impl is OK.

#[unstable(feature = "once_cell", issue = "68198")]
impl<T, F: RefUnwindSafe> RefUnwindSafe for Lazy<T, F> where Once<T>: RefUnwindSafe {}

impl<T, F> Lazy<T, F> {
    /// Creates a new lazy value with the given initializing
    /// function.
    #[unstable(feature = "once_cell", issue = "68198")]
    pub const fn new(f: F) -> Lazy<T, F> {
        Lazy { cell: Once::new(), init: Cell::new(Some(f)) }
    }
}

impl<T, F: FnOnce() -> T> Lazy<T, F> {
    /// Forces the evaluation of this lazy value and
    /// returns a reference to result. This is equivalent
    /// to the `Deref` impl, but is explicit.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(once_cell)]
    ///
    /// use std::lazy::Lazy;
    ///
    /// let lazy = Lazy::new(|| 92);
    ///
    /// assert_eq!(Lazy::force(&lazy), &92);
    /// assert_eq!(&*lazy, &92);
    /// ```
    #[unstable(feature = "once_cell", issue = "68198")]
    pub fn force(this: &Lazy<T, F>) -> &T {
        this.cell.get_or_init(|| match this.init.take() {
            Some(f) => f(),
            None => panic!("Lazy instance has previously been poisoned"),
        })
    }
}

#[unstable(feature = "once_cell", issue = "68198")]
impl<T, F: FnOnce() -> T> Deref for Lazy<T, F> {
    type Target = T;
    fn deref(&self) -> &T {
        Lazy::force(self)
    }
}

#[unstable(feature = "once_cell", issue = "68198")]
impl<T: Default> Default for Lazy<T> {
    /// Creates a new lazy value using `Default` as the initializing function.
    fn default() -> Lazy<T> {
        Lazy::new(T::default)
    }
}
