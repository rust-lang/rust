use rustc_macros::{Decodable, Encodable};

use std::num::NonZeroU8;

use crate::vec::Idx;

#[derive(Eq, PartialEq, Hash, Clone, Copy, Decodable, Encodable, Debug)]
#[repr(transparent)]
struct NonMaxU8 {
    repr: NonZeroU8,
}

impl NonMaxU8 {
    #[inline]
    fn new(n: u8) -> Option<Self> {
        Some(Self { repr: NonZeroU8::new(n.wrapping_add(1))? })
    }

    #[inline]
    fn get(&self) -> u8 {
        self.repr.get().wrapping_sub(1)
    }
}

#[derive(Eq, PartialEq, Hash, Clone, Copy, Decodable, Encodable, Debug)]
#[repr(C)]
pub struct DenseBitSet {
    words: [u64; 3],
    tail: [u8; 7],
    domain_size: NonMaxU8,
}

const INLINE_CAPACITY_BYTES: usize = 31;
const INLINE_CAPACITY_BITS: usize = INLINE_CAPACITY_BYTES * 8;

impl DenseBitSet {
    #[inline]
    pub fn new_empty(domain_size: usize) -> Self {
        assert!(domain_size <= INLINE_CAPACITY_BITS);
        Self { words: [0; 3], tail: [0; 7], domain_size: NonMaxU8::new(domain_size as u8).unwrap() }
    }

    #[inline]
    pub fn domain_size(&self) -> usize {
        self.domain_size.get() as usize
    }

    #[inline]
    pub fn words(&self) -> &[u8] {
        let (words, domain_size) = self.raw_parts();
        let used_words = num_words(domain_size);
        &words[..used_words]
    }

    #[inline]
    pub fn words_mut(&mut self) -> &mut [u8] {
        let (words, domain_size) = self.raw_parts_mut();
        let used_words = num_words(domain_size);
        &mut words[..used_words]
    }

    #[inline]
    pub fn raw_parts(&self) -> (&[u8], usize) {
        let words = unsafe {
            std::slice::from_raw_parts(self as *const Self as *const u8, INLINE_CAPACITY_BYTES)
        };
        (words, self.domain_size.get() as usize)
    }

    #[inline]
    pub fn raw_parts_mut(&mut self) -> (&mut [u8], usize) {
        let words = unsafe {
            std::slice::from_raw_parts_mut(self as *mut Self as *mut u8, INLINE_CAPACITY_BYTES)
        };
        (words, self.domain_size.get() as usize)
    }

    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = u64> + '_ {
        DenseBitSetIter { state: 0, set: self }
    }

    #[inline]
    fn as_u64s(&self) -> &[u64; 4] {
        unsafe { &*(self as *const Self as *const [u64; 4]) }
    }

    #[inline]
    fn as_u64s_mut(&mut self) -> &mut [u64; 4] {
        unsafe { &mut *(self as *mut Self as *mut [u64; 4]) }
    }

    #[inline]
    pub fn union(&mut self, other: &Self) -> bool {
        // Since domain_size of self and other must be equal, we can just or all the bits, and the
        // domain size bits will not be changed.

        let slf = self.as_u64s_mut();
        let oth = other.as_u64s();

        let mut changed = 0;
        for i in 0..4 {
            let old = slf[i];
            let new = old | oth[i];
            slf[i] = new;
            changed |= old ^ new;
        }

        changed != 0
    }
}

pub struct DenseBitSetIter<'a> {
    state: usize,
    set: &'a DenseBitSet,
}

const STATE_TAIL: usize = 3;
const STATE_DONE: usize = 4;

impl Iterator for DenseBitSetIter<'_> {
    type Item = u64;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.state == STATE_DONE {
            return None;
        }
        // SAFETY: DenseBitSet is POD
        let word = unsafe {
            let mut word = *(self.set as *const DenseBitSet as *const u64).add(self.state);
            if self.state == STATE_TAIL {
                word &= u64::MAX >> 8;
            }
            word
        };
        self.state += 1;
        Some(word)
    }
}

#[inline]
fn num_words<T: Idx>(domain_size: T) -> usize {
    const WORD_BITS: usize = 8;
    (domain_size.index() + WORD_BITS - 1) / WORD_BITS
}
