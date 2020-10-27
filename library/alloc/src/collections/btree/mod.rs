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
/// XorShiftRng
struct DeterministicRng {
    x: u32,
    y: u32,
    z: u32,
    w: u32,
}

#[cfg(test)]
impl DeterministicRng {
    fn new() -> Self {
        DeterministicRng { x: 0x193a6754, y: 0xa8a7d469, z: 0x97830e05, w: 0x113ba7bb }
    }

    fn next(&mut self) -> u32 {
        let x = self.x;
        let t = x ^ (x << 11);
        self.x = self.y;
        self.y = self.z;
        self.z = self.w;
        let w_ = self.w;
        self.w = w_ ^ (w_ >> 19) ^ (t ^ (t >> 8));
        self.w
    }
}
