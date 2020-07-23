pub mod map;
mod navigate;
mod node;
mod search;
pub mod set;

#[doc(hidden)]
trait Recover<Q: ?Sized> {
    type Key;

    fn get(&self, key: &Q) -> Option<&Self::Key>;
    fn take(&mut self, key: &Q) -> Option<Self::Key>;
    fn replace(&mut self, key: Self::Key) -> Option<Self::Key>;
}

/// Same purpose as `Option::unwrap` but doesn't always guarantee a panic
/// if the option contains no value.
/// SAFETY: the caller must ensure that the option contains a value.
#[inline(always)]
pub unsafe fn unwrap_unchecked<T>(val: Option<T>) -> T {
    if cfg!(debug_assertions) {
        val.expect("'unchecked' unwrap on None in BTreeMap")
    } else {
        val.unwrap()
        // val.unwrap_or_else(|| unsafe { core::hint::unreachable_unchecked() })
        // ...is considerably slower
    }
}
