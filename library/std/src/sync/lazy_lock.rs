use crate::cell::UnsafeCell;
use crate::fmt;
use crate::mem::ManuallyDrop;
use crate::ops::Deref;
use crate::panic::{RefUnwindSafe, UnwindSafe};
use crate::sync::Once;

use super::once::ExclusiveState;

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
///
/// [`LazyCell`]: crate::cell::LazyCell
///
/// # Examples
///
/// ```
/// #![feature(lazy_cell)]
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
#[unstable(feature = "lazy_cell", issue = "109736")]
pub struct LazyLock<T, F = fn() -> T> {
    once: Once,
    data: UnsafeCell<Data<T, F>>,
}

impl<T, F: FnOnce() -> T> LazyLock<T, F> {
    /// Creates a new lazy value with the given initializing
    /// function.
    #[inline]
    #[unstable(feature = "lazy_cell", issue = "109736")]
    pub const fn new(f: F) -> LazyLock<T, F> {
        LazyLock { once: Once::new(), data: UnsafeCell::new(Data { f: ManuallyDrop::new(f) }) }
    }

    /// Forces the evaluation of this lazy value and
    /// returns a reference to result. This is equivalent
    /// to the `Deref` impl, but is explicit.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(lazy_cell)]
    ///
    /// use std::sync::LazyLock;
    ///
    /// let lazy = LazyLock::new(|| 92);
    ///
    /// assert_eq!(LazyLock::force(&lazy), &92);
    /// assert_eq!(&*lazy, &92);
    /// ```
    #[inline]
    #[unstable(feature = "lazy_cell", issue = "109736")]
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
    /// Get the inner value if it has already been initialized.
    fn get(&self) -> Option<&T> {
        if self.once.is_completed() {
            // SAFETY:
            // The closure has been run successfully, so `value` has been initialized
            // and will not be modified again.
            Some(unsafe { &*(*self.data.get()).value })
        } else {
            None
        }
    }
}

#[unstable(feature = "lazy_cell", issue = "109736")]
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

#[unstable(feature = "lazy_cell", issue = "109736")]
impl<T, F: FnOnce() -> T> Deref for LazyLock<T, F> {
    type Target = T;

    #[inline]
    fn deref(&self) -> &T {
        LazyLock::force(self)
    }
}

#[unstable(feature = "lazy_cell", issue = "109736")]
impl<T: Default> Default for LazyLock<T> {
    /// Creates a new lazy value using `Default` as the initializing function.
    #[inline]
    fn default() -> LazyLock<T> {
        LazyLock::new(T::default)
    }
}

#[unstable(feature = "lazy_cell", issue = "109736")]
impl<T: fmt::Debug, F> fmt::Debug for LazyLock<T, F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.get() {
            Some(v) => f.debug_tuple("LazyLock").field(v).finish(),
            None => f.write_str("LazyLock(Uninit)"),
        }
    }
}

// We never create a `&F` from a `&LazyLock<T, F>` so it is fine
// to not impl `Sync` for `F`
#[unstable(feature = "lazy_cell", issue = "109736")]
unsafe impl<T: Sync + Send, F: Send> Sync for LazyLock<T, F> {}
// auto-derived `Send` impl is OK.

#[unstable(feature = "lazy_cell", issue = "109736")]
impl<T: RefUnwindSafe + UnwindSafe, F: UnwindSafe> RefUnwindSafe for LazyLock<T, F> {}
#[unstable(feature = "lazy_cell", issue = "109736")]
impl<T: UnwindSafe, F: UnwindSafe> UnwindSafe for LazyLock<T, F> {}

#[cfg(test)]
mod tests;
