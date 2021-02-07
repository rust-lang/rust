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

#[cfg(test)]
/// XorShiftRng
struct DeterministicRng {
    count: usize,
    x: u32,
    y: u32,
    z: u32,
    w: u32,
}

#[cfg(test)]
impl DeterministicRng {
    fn new() -> Self {
        DeterministicRng { count: 0, x: 0x193a6754, y: 0xa8a7d469, z: 0x97830e05, w: 0x113ba7bb }
    }

    /// Guarantees that each returned number is unique.
    fn next(&mut self) -> u32 {
        self.count += 1;
        assert!(self.count <= 70029);
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
