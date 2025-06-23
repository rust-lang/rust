use super::poison::once::ExclusiveState;
use crate::cell::UnsafeCell;
use crate::mem::ManuallyDrop;
use crate::ops::{Deref, DerefMut};
use crate::panic::{RefUnwindSafe, UnwindSafe};
use crate::sync::Once;
use crate::{fmt, ptr};

// We use the state of a Once as discriminant value. Upon creation, the state is
// "incomplete" and `f` contains the initialization closure. In the first call to
// `call_once`, `f` is taken and run. If it succeeds, `value` is set and the state
// is changed to "complete". If it panics, the Once is poisoned, so none of the
// two fields is initialized.
union Data<T, F> {
    value: ManuallyDrop<T>,
    f: ManuallyDrop<F>,
}

/// A value which is initialized on the first access.
///
/// This type is a thread-safe [`LazyCell`], and can be used in statics.
/// Since initialization may be called from multiple threads, any
/// dereferencing call will block the calling thread if another
/// initialization routine is currently running.
///
/// [`LazyCell`]: crate::cell::LazyCell
///
/// # Examples
///
/// Initialize static variables with `LazyLock`.
/// ```
/// use std::sync::LazyLock;
///
/// // Note: static items do not call [`Drop`] on program termination, so this won't be deallocated.
/// // this is fine, as the OS can deallocate the terminated program faster than we can free memory
/// // but tools like valgrind might report "memory leaks" as it isn't obvious this is intentional.
/// static DEEP_THOUGHT: LazyLock<String> = LazyLock::new(|| {
/// # mod another_crate {
/// #     pub fn great_question() -> String { "42".to_string() }
/// # }
///     // M3 Ultra takes about 16 million years in --release config
///     another_crate::great_question()
/// });
///
/// // The `String` is built, stored in the `LazyLock`, and returned as `&String`.
/// let _ = &*DEEP_THOUGHT;
/// ```
///
/// Initialize fields with `LazyLock`.
/// ```
/// use std::sync::LazyLock;
///
/// #[derive(Debug)]
/// struct UseCellLock {
///     number: LazyLock<u32>,
/// }
/// fn main() {
///     let lock: LazyLock<u32> = LazyLock::new(|| 0u32);
///
///     let data = UseCellLock { number: lock };
///     println!("{}", *data.number);
/// }
/// ```
#[stable(feature = "lazy_cell", since = "1.80.0")]
pub struct LazyLock<T, F = fn() -> T> {
    // FIXME(nonpoison_once): if possible, switch to nonpoison version once it is available
    once: Once,
    data: UnsafeCell<Data<T, F>>,
}

impl<T, F: FnOnce() -> T> LazyLock<T, F> {
    /// Creates a new lazy value with the given initializing function.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::LazyLock;
    ///
    /// let hello = "Hello, World!".to_string();
    ///
    /// let lazy = LazyLock::new(|| hello.to_uppercase());
    ///
    /// assert_eq!(&*lazy, "HELLO, WORLD!");
    /// ```
    #[inline]
    #[stable(feature = "lazy_cell", since = "1.80.0")]
    #[rustc_const_stable(feature = "lazy_cell", since = "1.80.0")]
    pub const fn new(f: F) -> LazyLock<T, F> {
        LazyLock { once: Once::new(), data: UnsafeCell::new(Data { f: ManuallyDrop::new(f) }) }
    }

    /// Creates a new lazy value that is already initialized.
    #[inline]
    #[cfg(test)]
    pub(crate) fn preinit(value: T) -> LazyLock<T, F> {
        let once = Once::new();
        once.call_once(|| {});
        LazyLock { once, data: UnsafeCell::new(Data { value: ManuallyDrop::new(value) }) }
    }

    /// Consumes this `LazyLock` returning the stored value.
    ///
    /// Returns `Ok(value)` if `Lazy` is initialized and `Err(f)` otherwise.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(lazy_cell_into_inner)]
    ///
    /// use std::sync::LazyLock;
    ///
    /// let hello = "Hello, World!".to_string();
    ///
    /// let lazy = LazyLock::new(|| hello.to_uppercase());
    ///
    /// assert_eq!(&*lazy, "HELLO, WORLD!");
    /// assert_eq!(LazyLock::into_inner(lazy).ok(), Some("HELLO, WORLD!".to_string()));
    /// ```
    #[unstable(feature = "lazy_cell_into_inner", issue = "125623")]
    pub fn into_inner(mut this: Self) -> Result<T, F> {
        let state = this.once.state();
        match state {
            ExclusiveState::Poisoned => panic_poisoned(),
            state => {
                let this = ManuallyDrop::new(this);
                let data = unsafe { ptr::read(&this.data) }.into_inner();
                match state {
                    ExclusiveState::Incomplete => Err(ManuallyDrop::into_inner(unsafe { data.f })),
                    ExclusiveState::Complete => Ok(ManuallyDrop::into_inner(unsafe { data.value })),
                    ExclusiveState::Poisoned => unreachable!(),
                }
            }
        }
    }

    /// Forces the evaluation of this lazy value and returns a mutable reference to
    /// the result.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(lazy_get)]
    /// use std::sync::LazyLock;
    ///
    /// let mut lazy = LazyLock::new(|| 92);
    ///
    /// let p = LazyLock::force_mut(&mut lazy);
    /// assert_eq!(*p, 92);
    /// *p = 44;
    /// assert_eq!(*lazy, 44);
    /// ```
    #[inline]
    #[unstable(feature = "lazy_get", issue = "129333")]
    pub fn force_mut(this: &mut LazyLock<T, F>) -> &mut T {
        #[cold]
        /// # Safety
        /// May only be called when the state is `Incomplete`.
        unsafe fn really_init_mut<T, F: FnOnce() -> T>(this: &mut LazyLock<T, F>) -> &mut T {
            struct PoisonOnPanic<'a, T, F>(&'a mut LazyLock<T, F>);
            impl<T, F> Drop for PoisonOnPanic<'_, T, F> {
                #[inline]
                fn drop(&mut self) {
                    self.0.once.set_state(ExclusiveState::Poisoned);
                }
            }

            // SAFETY: We always poison if the initializer panics (then we never check the data),
            // or set the data on success.
            let f = unsafe { ManuallyDrop::take(&mut this.data.get_mut().f) };
            // INVARIANT: Initiated from mutable reference, don't drop because we read it.
            let guard = PoisonOnPanic(this);
            let data = f();
            guard.0.data.get_mut().value = ManuallyDrop::new(data);
            guard.0.once.set_state(ExclusiveState::Complete);
            core::mem::forget(guard);
            // SAFETY: We put the value there above.
            unsafe { &mut this.data.get_mut().value }
        }

        let state = this.once.state();
        match state {
            ExclusiveState::Poisoned => panic_poisoned(),
            // SAFETY: The `Once` states we completed the initialization.
            ExclusiveState::Complete => unsafe { &mut this.data.get_mut().value },
            // SAFETY: The state is `Incomplete`.
            ExclusiveState::Incomplete => unsafe { really_init_mut(this) },
        }
    }

    /// Forces the evaluation of this lazy value and returns a reference to
    /// result. This is equivalent to the `Deref` impl, but is explicit.
    ///
    /// This method will block the calling thread if another initialization
    /// routine is currently running.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::LazyLock;
    ///
    /// let lazy = LazyLock::new(|| 92);
    ///
    /// assert_eq!(LazyLock::force(&lazy), &92);
    /// assert_eq!(&*lazy, &92);
    /// ```
    #[inline]
    #[stable(feature = "lazy_cell", since = "1.80.0")]
    pub fn force(this: &LazyLock<T, F>) -> &T {
        this.once.call_once(|| {
            // SAFETY: `call_once` only runs this closure once, ever.
            let data = unsafe { &mut *this.data.get() };
            let f = unsafe { ManuallyDrop::take(&mut data.f) };
            let value = f();
            data.value = ManuallyDrop::new(value);
        });

        // SAFETY:
        // There are four possible scenarios:
        // * the closure was called and initialized `value`.
        // * the closure was called and panicked, so this point is never reached.
        // * the closure was not called, but a previous call initialized `value`.
        // * the closure was not called because the Once is poisoned, so this point
        //   is never reached.
        // So `value` has definitely been initialized and will not be modified again.
        unsafe { &*(*this.data.get()).value }
    }
}

impl<T, F> LazyLock<T, F> {
    /// Returns a mutable reference to the value if initialized, or `None` if not.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(lazy_get)]
    ///
    /// use std::sync::LazyLock;
    ///
    /// let mut lazy = LazyLock::new(|| 92);
    ///
    /// assert_eq!(LazyLock::get_mut(&mut lazy), None);
    /// let _ = LazyLock::force(&lazy);
    /// *LazyLock::get_mut(&mut lazy).unwrap() = 44;
    /// assert_eq!(*lazy, 44);
    /// ```
    #[inline]
    #[unstable(feature = "lazy_get", issue = "129333")]
    pub fn get_mut(this: &mut LazyLock<T, F>) -> Option<&mut T> {
        // `state()` does not perform an atomic load, so prefer it over `is_complete()`.
        let state = this.once.state();
        match state {
            // SAFETY:
            // The closure has been run successfully, so `value` has been initialized.
            ExclusiveState::Complete => Some(unsafe { &mut this.data.get_mut().value }),
            _ => None,
        }
    }

    /// Returns a reference to the value if initialized, or `None` if not.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(lazy_get)]
    ///
    /// use std::sync::LazyLock;
    ///
    /// let lazy = LazyLock::new(|| 92);
    ///
    /// assert_eq!(LazyLock::get(&lazy), None);
    /// let _ = LazyLock::force(&lazy);
    /// assert_eq!(LazyLock::get(&lazy), Some(&92));
    /// ```
    #[inline]
    #[unstable(feature = "lazy_get", issue = "129333")]
    pub fn get(this: &LazyLock<T, F>) -> Option<&T> {
        if this.once.is_completed() {
            // SAFETY:
            // The closure has been run successfully, so `value` has been initialized
            // and will not be modified again.
            Some(unsafe { &(*this.data.get()).value })
        } else {
            None
        }
    }
}

#[stable(feature = "lazy_cell", since = "1.80.0")]
impl<T, F> Drop for LazyLock<T, F> {
    fn drop(&mut self) {
        match self.once.state() {
            ExclusiveState::Incomplete => unsafe { ManuallyDrop::drop(&mut self.data.get_mut().f) },
            ExclusiveState::Complete => unsafe {
                ManuallyDrop::drop(&mut self.data.get_mut().value)
            },
            ExclusiveState::Poisoned => {}
        }
    }
}

#[stable(feature = "lazy_cell", since = "1.80.0")]
impl<T, F: FnOnce() -> T> Deref for LazyLock<T, F> {
    type Target = T;

    /// Dereferences the value.
    ///
    /// This method will block the calling thread if another initialization
    /// routine is currently running.
    ///
    #[inline]
    fn deref(&self) -> &T {
        LazyLock::force(self)
    }
}

#[stable(feature = "lazy_deref_mut", since = "1.89.0")]
impl<T, F: FnOnce() -> T> DerefMut for LazyLock<T, F> {
    #[inline]
    fn deref_mut(&mut self) -> &mut T {
        LazyLock::force_mut(self)
    }
}

#[stable(feature = "lazy_cell", since = "1.80.0")]
impl<T: Default> Default for LazyLock<T> {
    /// Creates a new lazy value using `Default` as the initializing function.
    #[inline]
    fn default() -> LazyLock<T> {
        LazyLock::new(T::default)
    }
}

#[stable(feature = "lazy_cell", since = "1.80.0")]
impl<T: fmt::Debug, F> fmt::Debug for LazyLock<T, F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut d = f.debug_tuple("LazyLock");
        match LazyLock::get(self) {
            Some(v) => d.field(v),
            None => d.field(&format_args!("<uninit>")),
        };
        d.finish()
    }
}

#[cold]
#[inline(never)]
fn panic_poisoned() -> ! {
    panic!("LazyLock instance has previously been poisoned")
}

// We never create a `&F` from a `&LazyLock<T, F>` so it is fine
// to not impl `Sync` for `F`.
#[stable(feature = "lazy_cell", since = "1.80.0")]
unsafe impl<T: Sync + Send, F: Send> Sync for LazyLock<T, F> {}
// auto-derived `Send` impl is OK.

#[stable(feature = "lazy_cell", since = "1.80.0")]
impl<T: RefUnwindSafe + UnwindSafe, F: UnwindSafe> RefUnwindSafe for LazyLock<T, F> {}
#[stable(feature = "lazy_cell", since = "1.80.0")]
impl<T: UnwindSafe, F: UnwindSafe> UnwindSafe for LazyLock<T, F> {}
