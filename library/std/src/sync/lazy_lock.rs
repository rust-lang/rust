use crate::cell::Cell;
use crate::fmt;
use crate::ops::Deref;
use crate::panic::{RefUnwindSafe, UnwindSafe};
use crate::sync::OnceLock;

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
/// use std::sync::LazyLock;
///
/// static HASHMAP: LazyLock<HashMap<i32, String>> = LazyLock::new(|| {
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
pub struct LazyLock<T, F = fn() -> T> {
    cell: OnceLock<T>,
    init: Cell<Option<F>>,
}

impl<T, F> LazyLock<T, F> {
    /// Creates a new lazy value with the given initializing
    /// function.
    #[unstable(feature = "once_cell", issue = "74465")]
    pub const fn new(f: F) -> LazyLock<T, F> {
        LazyLock { cell: OnceLock::new(), init: Cell::new(Some(f)) }
    }
}

impl<T, F: FnOnce() -> T> LazyLock<T, F> {
    /// Forces the evaluation of this lazy value and
    /// returns a reference to result. This is equivalent
    /// to the `Deref` impl, but is explicit.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(once_cell)]
    ///
    /// use std::sync::LazyLock;
    ///
    /// let lazy = LazyLock::new(|| 92);
    ///
    /// assert_eq!(LazyLock::force(&lazy), &92);
    /// assert_eq!(&*lazy, &92);
    /// ```
    #[unstable(feature = "once_cell", issue = "74465")]
    pub fn force(this: &LazyLock<T, F>) -> &T {
        this.cell.get_or_init(|| match this.init.take() {
            Some(f) => f(),
            None => panic!("Lazy instance has previously been poisoned"),
        })
    }
}

#[unstable(feature = "once_cell", issue = "74465")]
impl<T, F: FnOnce() -> T> Deref for LazyLock<T, F> {
    type Target = T;
    fn deref(&self) -> &T {
        LazyLock::force(self)
    }
}

#[unstable(feature = "once_cell", issue = "74465")]
impl<T: Default> Default for LazyLock<T> {
    /// Creates a new lazy value using `Default` as the initializing function.
    fn default() -> LazyLock<T> {
        LazyLock::new(T::default)
    }
}

#[unstable(feature = "once_cell", issue = "74465")]
impl<T: fmt::Debug, F> fmt::Debug for LazyLock<T, F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Lazy").field("cell", &self.cell).finish_non_exhaustive()
    }
}

// We never create a `&F` from a `&LazyLock<T, F>` so it is fine
// to not impl `Sync` for `F`
// we do create a `&mut Option<F>` in `force`, but this is
// properly synchronized, so it only happens once
// so it also does not contribute to this impl.
#[unstable(feature = "once_cell", issue = "74465")]
unsafe impl<T, F: Send> Sync for LazyLock<T, F> where OnceLock<T>: Sync {}
// auto-derived `Send` impl is OK.

#[unstable(feature = "once_cell", issue = "74465")]
impl<T, F: UnwindSafe> RefUnwindSafe for LazyLock<T, F> where OnceLock<T>: RefUnwindSafe {}
#[unstable(feature = "once_cell", issue = "74465")]
impl<T, F: UnwindSafe> UnwindSafe for LazyLock<T, F> where OnceLock<T>: UnwindSafe {}

#[cfg(test)]
mod tests;
