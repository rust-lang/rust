//! A none hashing [`Hasher`] implementation.
use std::{
    hash::{BuildHasher, Hasher},
    marker::PhantomData,
};

pub type NoHashHashMap<K, V> = std::collections::HashMap<K, V, NoHashHasherBuilder<K>>;
pub type NoHashHashSet<K> = std::collections::HashSet<K, NoHashHasherBuilder<K>>;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct NoHashHasherBuilder<T>(PhantomData<T>);

impl<T> Default for NoHashHasherBuilder<T> {
    fn default() -> Self {
        Self(Default::default())
    }
}

pub trait NoHashHashable {}
impl NoHashHashable for usize {}
impl NoHashHashable for u32 {}

pub struct NoHashHasher(u64);

impl<T: NoHashHashable> BuildHasher for NoHashHasherBuilder<T> {
    type Hasher = NoHashHasher;
    fn build_hasher(&self) -> Self::Hasher {
        NoHashHasher(0)
    }
}

impl Hasher for NoHashHasher {
    fn finish(&self) -> u64 {
        self.0
    }

    fn write(&mut self, _: &[u8]) {
        unimplemented!("NoHashHasher should only be used for hashing primitive integers")
    }

    fn write_u8(&mut self, i: u8) {
        self.0 = i as u64;
    }

    fn write_u16(&mut self, i: u16) {
        self.0 = i as u64;
    }

    fn write_u32(&mut self, i: u32) {
        self.0 = i as u64;
    }

    fn write_u64(&mut self, i: u64) {
        self.0 = i as u64;
    }

    fn write_usize(&mut self, i: usize) {
        self.0 = i as u64;
    }

    fn write_i8(&mut self, i: i8) {
        self.0 = i as u64;
    }

    fn write_i16(&mut self, i: i16) {
        self.0 = i as u64;
    }

    fn write_i32(&mut self, i: i32) {
        self.0 = i as u64;
    }

    fn write_i64(&mut self, i: i64) {
        self.0 = i as u64;
    }

    fn write_isize(&mut self, i: isize) {
        self.0 = i as u64;
    }
}
