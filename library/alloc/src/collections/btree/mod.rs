mod append;
mod borrow;
pub mod map;
mod mem;
mod merge_iter;
mod navigate;
mod node;
mod remove;
mod search;
pub mod set;
mod split;

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
    val.unwrap_or_else(|| {
        if cfg!(debug_assertions) {
            panic!("'unchecked' unwrap on None in BTreeMap");
        } else {
            unsafe {
                core::intrinsics::unreachable();
            }
        }
    })
}

#[cfg(test)]
mod testing;
