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

#[inline(always)]
pub unsafe fn unwrap_unchecked<T>(val: Option<T>) -> T {
    val.unwrap_or_else(|| {
        if cfg!(debug_assertions) {
            panic!("'unchecked' unwrap on None in BTreeMap");
        } else {
            core::intrinsics::unreachable();
        }
    })
}
