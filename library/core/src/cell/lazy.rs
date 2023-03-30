use crate::cell::{Cell, OnceCell};
use crate::fmt;
use crate::ops::Deref;

/// A value which is initialized on the first access.
///
/// For a thread-safe version of this struct, see [`std::sync::LazyLock`].
///
/// [`std::sync::LazyLock`]: ../../std/sync/struct.LazyLock.html
///
/// # Examples
///
/// ```
/// #![feature(lazy_cell)]
///
/// use std::cell::LazyCell;
///
/// let lazy: LazyCell<i32> = LazyCell::new(|| {
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
#[unstable(feature = "lazy_cell", issue = "109736")]
pub struct LazyCell<T, F = fn() -> T> {
    cell: OnceCell<T>,
    init: Cell<Option<F>>,
}

impl<T, F: FnOnce() -> T> LazyCell<T, F> {
    /// Creates a new lazy value with the given initializing function.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(lazy_cell)]
    ///
    /// use std::cell::LazyCell;
    ///
    /// let hello = "Hello, World!".to_string();
    ///
    /// let lazy = LazyCell::new(|| hello.to_uppercase());
    ///
    /// assert_eq!(&*lazy, "HELLO, WORLD!");
    /// ```
    #[inline]
    #[unstable(feature = "lazy_cell", issue = "109736")]
    pub const fn new(init: F) -> LazyCell<T, F> {
        LazyCell { cell: OnceCell::new(), init: Cell::new(Some(init)) }
    }

    /// Forces the evaluation of this lazy value and returns a reference to
    /// the result.
    ///
    /// This is equivalent to the `Deref` impl, but is explicit.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(lazy_cell)]
    ///
    /// use std::cell::LazyCell;
    ///
    /// let lazy = LazyCell::new(|| 92);
    ///
    /// assert_eq!(LazyCell::force(&lazy), &92);
    /// assert_eq!(&*lazy, &92);
    /// ```
    #[inline]
    #[unstable(feature = "lazy_cell", issue = "109736")]
    pub fn force(this: &LazyCell<T, F>) -> &T {
        this.cell.get_or_init(|| match this.init.take() {
            Some(f) => f(),
            None => panic!("`Lazy` instance has previously been poisoned"),
        })
    }
}

#[unstable(feature = "lazy_cell", issue = "109736")]
impl<T, F: FnOnce() -> T> Deref for LazyCell<T, F> {
    type Target = T;
    #[inline]
    fn deref(&self) -> &T {
        LazyCell::force(self)
    }
}

#[unstable(feature = "lazy_cell", issue = "109736")]
impl<T: Default> Default for LazyCell<T> {
    /// Creates a new lazy value using `Default` as the initializing function.
    #[inline]
    fn default() -> LazyCell<T> {
        LazyCell::new(T::default)
    }
}

#[unstable(feature = "lazy_cell", issue = "109736")]
impl<T: fmt::Debug, F> fmt::Debug for LazyCell<T, F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Lazy").field("cell", &self.cell).field("init", &"..").finish()
    }
}
