//! A non-hashing [`Hasher`] implementation.

#![deny(clippy::pedantic, missing_debug_implementations, missing_docs, rust_2018_idioms)]

use std::{
    hash::{BuildHasher, Hasher},
    marker::PhantomData,
};

/// A [`std::collections::HashMap`] with [`NoHashHasherBuilder`].
pub type NoHashHashMap<K, V> = std::collections::HashMap<K, V, NoHashHasherBuilder<K>>;

/// A [`std::collections::HashSet`] with [`NoHashHasherBuilder`].
pub type NoHashHashSet<K> = std::collections::HashSet<K, NoHashHasherBuilder<K>>;

/// A hasher builder for [`NoHashHasher`].
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct NoHashHasherBuilder<T>(PhantomData<T>);

impl<T> Default for NoHashHasherBuilder<T> {
    fn default() -> Self {
        Self(PhantomData)
    }
}

/// Types for which an acceptable hash function is to return itself.
///
/// This trait is implemented by sufficiently-small integer types. It should only be implemented for
/// foreign types that are newtypes of these types. If it is implemented on more complex types,
/// hashing will panic.
pub trait NoHashHashable {}

impl NoHashHashable for u8 {}
impl NoHashHashable for u16 {}
impl NoHashHashable for u32 {}
impl NoHashHashable for u64 {}
impl NoHashHashable for usize {}

impl NoHashHashable for i8 {}
impl NoHashHashable for i16 {}
impl NoHashHashable for i32 {}
impl NoHashHashable for i64 {}
impl NoHashHashable for isize {}

/// A hasher for [`NoHashHashable`] types.
#[derive(Debug)]
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
        unimplemented!("NoHashHasher should only be used for hashing sufficiently-small integer types and their newtypes")
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
        self.0 = i;
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
