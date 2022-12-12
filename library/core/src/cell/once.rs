use crate::cell::UnsafeCell;
use crate::fmt;
use crate::mem;

/// A cell which can be written to only once.
///
/// This allows obtaining a shared `&T` reference to its inner value without copying or replacing
/// it (unlike [`Cell`]), and without runtime borrow checks (unlike [`RefCell`]). However,
/// only immutable references can be obtained unless one has a mutable reference to the cell
/// itself.
///
/// For a thread-safe version of this struct, see [`std::sync::OnceLock`].
///
/// [`RefCell`]: crate::cell::RefCell
/// [`Cell`]: crate::cell::Cell
/// [`std::sync::OnceLock`]: ../../std/sync/struct.OnceLock.html
///
/// # Examples
///
/// ```
/// use std::cell::OnceCell;
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
#[stable(feature = "once_cell", since = "CURRENT_RUSTC_VERSION")]
pub struct OnceCell<T> {
    // Invariant: written to at most once.
    inner: UnsafeCell<Option<T>>,
}

impl<T> OnceCell<T> {
    /// Creates a new empty cell.
    #[inline]
    #[must_use]
    #[stable(feature = "once_cell", since = "CURRENT_RUSTC_VERSION")]
    #[rustc_const_stable(feature = "once_cell", since = "CURRENT_RUSTC_VERSION")]
    pub const fn new() -> OnceCell<T> {
        OnceCell { inner: UnsafeCell::new(None) }
    }

    /// Gets the reference to the underlying value.
    ///
    /// Returns `None` if the cell is empty.
    #[inline]
    #[stable(feature = "once_cell", since = "CURRENT_RUSTC_VERSION")]
    pub fn get(&self) -> Option<&T> {
        // SAFETY: Safe due to `inner`'s invariant
        unsafe { &*self.inner.get() }.as_ref()
    }

    /// Gets the mutable reference to the underlying value.
    ///
    /// Returns `None` if the cell is empty.
    #[inline]
    #[stable(feature = "once_cell", since = "CURRENT_RUSTC_VERSION")]
    pub fn get_mut(&mut self) -> Option<&mut T> {
        self.inner.get_mut().as_mut()
    }

    /// Sets the contents of the cell to `value`.
    ///
    /// # Errors
    ///
    /// This method returns `Ok(())` if the cell was empty and `Err(value)` if
    /// it was full.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::cell::OnceCell;
    ///
    /// let cell = OnceCell::new();
    /// assert!(cell.get().is_none());
    ///
    /// assert_eq!(cell.set(92), Ok(()));
    /// assert_eq!(cell.set(62), Err(62));
    ///
    /// assert!(cell.get().is_some());
    /// ```
    #[inline]
    #[stable(feature = "once_cell", since = "CURRENT_RUSTC_VERSION")]
    pub fn set(&self, value: T) -> Result<(), T> {
        // SAFETY: Safe because we cannot have overlapping mutable borrows
        let slot = unsafe { &*self.inner.get() };
        if slot.is_some() {
            return Err(value);
        }

        // SAFETY: This is the only place where we set the slot, no races
        // due to reentrancy/concurrency are possible, and we've
        // checked that slot is currently `None`, so this write
        // maintains the `inner`'s invariant.
        let slot = unsafe { &mut *self.inner.get() };
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
    /// # Examples
    ///
    /// ```
    /// use std::cell::OnceCell;
    ///
    /// let cell = OnceCell::new();
    /// let value = cell.get_or_init(|| 92);
    /// assert_eq!(value, &92);
    /// let value = cell.get_or_init(|| unreachable!());
    /// assert_eq!(value, &92);
    /// ```
    #[inline]
    #[stable(feature = "once_cell", since = "CURRENT_RUSTC_VERSION")]
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
    /// # Examples
    ///
    /// ```
    /// #![feature(once_cell_try)]
    ///
    /// use std::cell::OnceCell;
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
    #[unstable(feature = "once_cell_try", issue = "109737")]
    pub fn get_or_try_init<F, E>(&self, f: F) -> Result<&T, E>
    where
        F: FnOnce() -> Result<T, E>,
    {
        if let Some(val) = self.get() {
            return Ok(val);
        }
        /// Avoid inlining the initialization closure into the common path that fetches
        /// the already initialized value
        #[cold]
        fn outlined_call<F, T, E>(f: F) -> Result<T, E>
        where
            F: FnOnce() -> Result<T, E>,
        {
            f()
        }
        let val = outlined_call(f)?;
        // Note that *some* forms of reentrant initialization might lead to
        // UB (see `reentrant_init` test). I believe that just removing this
        // `assert`, while keeping `set/get` would be sound, but it seems
        // better to panic, rather than to silently use an old value.
        assert!(self.set(val).is_ok(), "reentrant init");
        Ok(self.get().unwrap())
    }

    /// Consumes the cell, returning the wrapped value.
    ///
    /// Returns `None` if the cell was empty.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::cell::OnceCell;
    ///
    /// let cell: OnceCell<String> = OnceCell::new();
    /// assert_eq!(cell.into_inner(), None);
    ///
    /// let cell = OnceCell::new();
    /// cell.set("hello".to_string()).unwrap();
    /// assert_eq!(cell.into_inner(), Some("hello".to_string()));
    /// ```
    #[inline]
    #[stable(feature = "once_cell", since = "CURRENT_RUSTC_VERSION")]
    pub fn into_inner(self) -> Option<T> {
        // Because `into_inner` takes `self` by value, the compiler statically verifies
        // that it is not currently borrowed. So it is safe to move out `Option<T>`.
        self.inner.into_inner()
    }

    /// Takes the value out of this `OnceCell`, moving it back to an uninitialized state.
    ///
    /// Has no effect and returns `None` if the `OnceCell` hasn't been initialized.
    ///
    /// Safety is guaranteed by requiring a mutable reference.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::cell::OnceCell;
    ///
    /// let mut cell: OnceCell<String> = OnceCell::new();
    /// assert_eq!(cell.take(), None);
    ///
    /// let mut cell = OnceCell::new();
    /// cell.set("hello".to_string()).unwrap();
    /// assert_eq!(cell.take(), Some("hello".to_string()));
    /// assert_eq!(cell.get(), None);
    /// ```
    #[inline]
    #[stable(feature = "once_cell", since = "CURRENT_RUSTC_VERSION")]
    pub fn take(&mut self) -> Option<T> {
        mem::take(self).into_inner()
    }
}

#[stable(feature = "once_cell", since = "CURRENT_RUSTC_VERSION")]
impl<T> Default for OnceCell<T> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

#[stable(feature = "once_cell", since = "CURRENT_RUSTC_VERSION")]
impl<T: fmt::Debug> fmt::Debug for OnceCell<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.get() {
            Some(v) => f.debug_tuple("OnceCell").field(v).finish(),
            None => f.write_str("OnceCell(Uninit)"),
        }
    }
}

#[stable(feature = "once_cell", since = "CURRENT_RUSTC_VERSION")]
impl<T: Clone> Clone for OnceCell<T> {
    #[inline]
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

#[stable(feature = "once_cell", since = "CURRENT_RUSTC_VERSION")]
impl<T: PartialEq> PartialEq for OnceCell<T> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.get() == other.get()
    }
}

#[stable(feature = "once_cell", since = "CURRENT_RUSTC_VERSION")]
impl<T: Eq> Eq for OnceCell<T> {}

#[stable(feature = "once_cell", since = "CURRENT_RUSTC_VERSION")]
#[rustc_const_unstable(feature = "const_convert", issue = "88674")]
impl<T> const From<T> for OnceCell<T> {
    /// Creates a new `OnceCell<T>` which already contains the given `value`.
    #[inline]
    fn from(value: T) -> Self {
        OnceCell { inner: UnsafeCell::new(Some(value)) }
    }
}

// Just like for `Cell<T>` this isn't needed, but results in nicer error messages.
#[stable(feature = "once_cell", since = "CURRENT_RUSTC_VERSION")]
impl<T> !Sync for OnceCell<T> {}
