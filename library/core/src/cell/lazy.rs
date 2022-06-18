use crate::cell::{Cell, OnceCell};
use crate::fmt;
use crate::ops::Deref;

/// A value which is initialized on the first access.
///
/// # Examples
///
/// ```
/// #![feature(once_cell)]
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
#[unstable(feature = "once_cell", issue = "74465")]
pub struct LazyCell<T, F = fn() -> T> {
    cell: OnceCell<T>,
    init: Cell<Option<F>>,
}

impl<T, F> LazyCell<T, F> {
    /// Creates a new lazy value with the given initializing function.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(once_cell)]
    ///
    /// use std::cell::LazyCell;
    ///
    /// let hello = "Hello, World!".to_string();
    ///
    /// let lazy = LazyCell::new(|| hello.to_uppercase());
    ///
    /// assert_eq!(&*lazy, "HELLO, WORLD!");
    /// ```
    #[unstable(feature = "once_cell", issue = "74465")]
    pub const fn new(init: F) -> LazyCell<T, F> {
        LazyCell { cell: OnceCell::new(), init: Cell::new(Some(init)) }
    }
}

impl<T, F: FnOnce() -> T> LazyCell<T, F> {
    /// Forces the evaluation of this lazy value and returns a reference to
    /// the result.
    ///
    /// This is equivalent to the `Deref` impl, but is explicit.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(once_cell)]
    ///
    /// use std::cell::LazyCell;
    ///
    /// let lazy = LazyCell::new(|| 92);
    ///
    /// assert_eq!(LazyCell::force(&lazy), &92);
    /// assert_eq!(&*lazy, &92);
    /// ```
    #[unstable(feature = "once_cell", issue = "74465")]
    pub fn force(this: &LazyCell<T, F>) -> &T {
        this.cell.get_or_init(|| match this.init.take() {
            Some(f) => f(),
            None => panic!("`Lazy` instance has previously been poisoned"),
        })
    }
}

#[unstable(feature = "once_cell", issue = "74465")]
impl<T, F: FnOnce() -> T> Deref for LazyCell<T, F> {
    type Target = T;
    fn deref(&self) -> &T {
        LazyCell::force(self)
    }
}

#[unstable(feature = "once_cell", issue = "74465")]
impl<T: Default> Default for LazyCell<T> {
    /// Creates a new lazy value using `Default` as the initializing function.
    fn default() -> LazyCell<T> {
        LazyCell::new(T::default)
    }
}

#[unstable(feature = "once_cell", issue = "74465")]
impl<T: fmt::Debug, F> fmt::Debug for LazyCell<T, F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Lazy").field("cell", &self.cell).field("init", &"..").finish()
    }
}
