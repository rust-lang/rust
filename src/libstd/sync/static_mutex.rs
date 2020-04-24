use crate::fmt;

/// A mutual exclusion primitive that relies on static type information only
///
/// In some cases synchronization can be proven statically: whenever you hold an exclusive `&mut`
/// reference, the Rust type system ensures that no other part of the program can hold another
/// reference to the data. Therefore it is safe to access it even if the current thread obtained
/// this reference via a channel. Whenever this is the case, the overhead of allocating and locking
/// a [`Mutex`] can be avoided by using this static version.
///
/// One example where this is often applicable is [`Future`], which requires an exclusive reference
/// for its [`poll`] method: While a given `Future` implementation may not be safe to access by
/// multiple threads concurrently, the executor can only run the `Future` on one thread at any
/// given time, making it [`Sync`] in practice as long as the implementation is `Send`. You can
/// therefore use the static mutex to prove that your data structure is `Sync` even though it
/// contains such a `Future`.
///
/// # Example
///
/// ```
/// #![feature(static_mutex)]
///
/// use std::sync::StaticMutex;
/// use std::future::Future;
///
/// struct MyThing {
///     future: StaticMutex<Box<dyn Future<Output = String> + Send>>,
/// }
///
/// impl MyThing {
///     // all accesses to `self.future` now require an exclusive reference or ownership
/// }
///
/// fn assert_sync<T: Sync>() {}
///
/// assert_sync::<MyThing>();
/// ```
///
/// [`Mutex`]: struct.Mutex.html
/// [`Future`]: ../future/trait.Future.html
/// [`poll`]: ../future/trait.Future.html#method.poll
/// [`Sync`]: ../marker/trait.Sync.html
#[unstable(feature = "static_mutex", issue = "71521")]
#[repr(transparent)]
pub struct StaticMutex<T>(T);

impl<T> StaticMutex<T> {
    /// Creates a new static mutex containing the given value.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(static_mutex)]
    ///
    /// use std::sync::StaticMutex;
    ///
    /// let mutex = StaticMutex::new(42);
    /// ```
    #[unstable(feature = "static_mutex", issue = "71521")]
    pub fn new(value: T) -> Self {
        Self(value)
    }

    /// Acquires a reference to the protected value.
    ///
    /// This is safe because it requires an exclusive reference to the mutex. Therefore this method
    /// neither panics nor does it return an error. This is in contrast to [`Mutex::get_mut`] which
    /// returns an error if another thread panicked while holding the lock. It is not recommended
    /// to send an exclusive reference to a potentially damaged value to another thread for further
    /// processing.
    ///
    /// [`Mutex::get_mut`]: struct.Mutex.html#method.get_mut
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(static_mutex)]
    ///
    /// use std::sync::StaticMutex;
    ///
    /// let mut mutex = StaticMutex::new(42);
    /// let value = mutex.get_mut();
    /// *value = 0;
    /// assert_eq!(*mutex.get_mut(), 0);
    /// ```
    #[unstable(feature = "static_mutex", issue = "71521")]
    pub fn get_mut(&mut self) -> &mut T {
        &mut self.0
    }

    /// Consumes this mutex, returning the underlying data.
    ///
    /// This is safe because it requires ownership of the mutex, therefore this method will neither
    /// panic nor does it return an error. This is in contrast to [`Mutex::into_inner`] which
    /// returns an error if another thread panicked while holding the lock. It is not recommended
    /// to send an exclusive reference to a potentially damaged value to another thread for further
    /// processing.
    ///
    /// [`Mutex::into_inner`]: struct.Mutex.html#method.into_inner
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(static_mutex)]
    ///
    /// use std::sync::StaticMutex;
    ///
    /// let mut mutex = StaticMutex::new(42);
    /// assert_eq!(mutex.into_inner(), 42);
    /// ```
    #[unstable(feature = "static_mutex", issue = "71521")]
    pub fn into_inner(self) -> T {
        self.0
    }
}

// this is safe because the only operations permitted on this data structure require exclusive
// access or ownership
#[unstable(feature = "static_mutex", issue = "71521")]
unsafe impl<T: Send> Sync for StaticMutex<T> {}

#[unstable(feature = "static_mutex", issue = "71521")]
impl<T> fmt::Debug for StaticMutex<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // cannot access the wrapped value without an exclusive reference
        f.write_str("<StaticMutex>")
    }
}
