// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// FIXME(Gankro): BitVec and BitSet are very tightly coupled. Ideally (for
// maintenance), they should be in separate files/modules, with BitSet only
// using BitVec's public API. This will be hard for performance though, because
// `BitVec` will not want to leak its internal representation while its internal
// representation as `u32`s must be assumed for best performance.

// FIXME(tbu-): `BitVec`'s methods shouldn't be `union`, `intersection`, but
// rather `or` and `and`.

// (1) Be careful, most things can overflow here because the amount of bits in
//     memory can overflow `usize`.
// (2) Make sure that the underlying vector has no excess length:
//     E. g. `nbits == 16`, `storage.len() == 2` would be excess length,
//     because the last word isn't used at all. This is important because some
//     methods rely on it (for *CORRECTNESS*).
// (3) Make sure that the unused bits in the last word are zeroed out, again
//     other methods rely on it for *CORRECTNESS*.
// (4) `BitSet` is tightly coupled with `BitVec`, so any changes you make in
// `BitVec` will need to be reflected in `BitSet`.

//! Collections implemented with bit vectors.
//!
//! # Examples
//!
//! This is a simple example of the [Sieve of Eratosthenes][sieve]
//! which calculates prime numbers up to a given limit.
//!
//! [sieve]: http://en.wikipedia.org/wiki/Sieve_of_Eratosthenes
//!
//! ```
//! # #![feature(collections, core, step_by)]
//! use std::collections::{BitSet, BitVec};
//! use std::num::Float;
//! use std::iter;
//!
//! let max_prime = 10000;
//!
//! // Store the primes as a BitSet
//! let primes = {
//!     // Assume all numbers are prime to begin, and then we
//!     // cross off non-primes progressively
//!     let mut bv = BitVec::from_elem(max_prime, true);
//!
//!     // Neither 0 nor 1 are prime
//!     bv.set(0, false);
//!     bv.set(1, false);
//!
//!     for i in iter::range_inclusive(2, (max_prime as f64).sqrt() as usize) {
//!         // if i is a prime
//!         if bv[i] {
//!             // Mark all multiples of i as non-prime (any multiples below i * i
//!             // will have been marked as non-prime previously)
//!             for j in (i * i..max_prime).step_by(i) { bv.set(j, false) }
//!         }
//!     }
//!     BitSet::from_bit_vec(bv)
//! };
//!
//! // Simple primality tests below our max bound
//! let print_primes = 20;
//! print!("The primes below {} are: ", print_primes);
//! for x in 0..print_primes {
//!     if primes.contains(&x) {
//!         print!("{} ", x);
//!     }
//! }
//! println!("");
//!
//! // We can manipulate the internal BitVec
//! let num_primes = primes.get_ref().iter().filter(|x| *x).count();
//! println!("There are {} primes below {}", num_primes, max_prime);
//! ```

use core::prelude::*;

use core::cmp::Ordering;
use core::cmp;
use core::default::Default;
use core::fmt;
use core::hash;
use core::iter::RandomAccessIterator;
use core::iter::{Chain, Enumerate, Repeat, Skip, Take, repeat, Cloned};
use core::iter::{self, FromIterator, IntoIterator};
use core::ops::Index;
use core::slice;
use core::{u8, u32, usize};
use bit_set; //so meta

use Vec;

type Blocks<'a> = Cloned<slice::Iter<'a, u32>>;
type MutBlocks<'a> = slice::IterMut<'a, u32>;
type MatchWords<'a> = Chain<Enumerate<Blocks<'a>>, Skip<Take<Enumerate<Repeat<u32>>>>>;

fn reverse_bits(byte: u8) -> u8 {
    let mut result = 0;
    for i in 0..u8::BITS {
        result |= ((byte >> i) & 1) << (u8::BITS - 1 - i);
    }
    result
}

// Take two BitVec's, and return iterators of their words, where the shorter one
// has been padded with 0's
fn match_words <'a,'b>(a: &'a BitVec, b: &'b BitVec) -> (MatchWords<'a>, MatchWords<'b>) {
    let a_len = a.storage.len();
    let b_len = b.storage.len();

    // have to uselessly pretend to pad the longer one for type matching
    if a_len < b_len {
        (a.blocks().enumerate().chain(iter::repeat(0).enumerate().take(b_len).skip(a_len)),
         b.blocks().enumerate().chain(iter::repeat(0).enumerate().take(0).skip(0)))
    } else {
        (a.blocks().enumerate().chain(iter::repeat(0).enumerate().take(0).skip(0)),
         b.blocks().enumerate().chain(iter::repeat(0).enumerate().take(a_len).skip(b_len)))
    }
}

static TRUE: bool = true;
static FALSE: bool = false;

/// The bitvector type.
///
/// # Examples
///
/// ```
/// # #![feature(collections)]
/// use std::collections::BitVec;
///
/// let mut bv = BitVec::from_elem(10, false);
///
/// // insert all primes less than 10
/// bv.set(2, true);
/// bv.set(3, true);
/// bv.set(5, true);
/// bv.set(7, true);
/// println!("{:?}", bv);
/// println!("total bits set to true: {}", bv.iter().filter(|x| *x).count());
///
/// // flip all values in bitvector, producing non-primes less than 10
/// bv.negate();
/// println!("{:?}", bv);
/// println!("total bits set to true: {}", bv.iter().filter(|x| *x).count());
///
/// // reset bitvector to empty
/// bv.clear();
/// println!("{:?}", bv);
/// println!("total bits set to true: {}", bv.iter().filter(|x| *x).count());
/// ```
#[unstable(feature = "collections",
           reason = "RFC 509")]
pub struct BitVec {
    /// Internal representation of the bit vector
    storage: Vec<u32>,
    /// The number of valid bits in the internal representation
    nbits: usize
}

// FIXME(Gankro): NopeNopeNopeNopeNope (wait for IndexGet to be a thing)
impl Index<usize> for BitVec {
    type Output = bool;

    #[inline]
    fn index(&self, i: usize) -> &bool {
        if self.get(i).expect("index out of bounds") {
            &TRUE
        } else {
            &FALSE
        }
    }
}

/// Computes how many blocks are needed to store that many bits
fn blocks_for_bits(bits: usize) -> usize {
    // If we want 17 bits, dividing by 32 will produce 0. So we add 1 to make sure we
    // reserve enough. But if we want exactly a multiple of 32, this will actually allocate
    // one too many. So we need to check if that's the case. We can do that by computing if
    // bitwise AND by `32 - 1` is 0. But LLVM should be able to optimize the semantically
    // superior modulo operator on a power of two to this.
    //
    // Note that we can technically avoid this branch with the expression
    // `(nbits + u32::BITS - 1) / 32::BITS`, but if nbits is almost usize::MAX this will overflow.
    if bits % u32::BITS == 0 {
        bits / u32::BITS
    } else {
        bits / u32::BITS + 1
    }
}

/// Computes the bitmask for the final word of the vector
fn mask_for_bits(bits: usize) -> u32 {
    // Note especially that a perfect multiple of u32::BITS should mask all 1s.
    !0 >> (u32::BITS - bits % u32::BITS) % u32::BITS
}

impl BitVec {
    /// Applies the given operation to the blocks of self and other, and sets
    /// self to be the result. This relies on the caller not to corrupt the
    /// last word.
    #[inline]
    fn process<F>(&mut self, other: &BitVec, mut op: F) -> bool where F: FnMut(u32, u32) -> u32 {
        assert_eq!(self.len(), other.len());
        // This could theoretically be a `debug_assert!`.
        assert_eq!(self.storage.len(), other.storage.len());
        let mut changed = false;
        for (a, b) in self.blocks_mut().zip(other.blocks()) {
            let w = op(*a, b);
            if *a != w {
                changed = true;
                *a = w;
            }
        }
        changed
    }

    /// Iterator over mutable refs to  the underlying blocks of data.
    fn blocks_mut(&mut self) -> MutBlocks {
        // (2)
        self.storage.iter_mut()
    }

    /// Iterator over the underlying blocks of data
    fn blocks(&self) -> Blocks {
        // (2)
        self.storage.iter().cloned()
    }

    /// An operation might screw up the unused bits in the last block of the
    /// `BitVec`. As per (3), it's assumed to be all 0s. This method fixes it up.
    fn fix_last_block(&mut self) {
        let extra_bits = self.len() % u32::BITS;
        if extra_bits > 0 {
            let mask = (1 << extra_bits) - 1;
            let storage_len = self.storage.len();
            self.storage[storage_len - 1] &= mask;
        }
    }

    /// Creates an empty `BitVec`.
    ///
    /// # Examples
    ///
    /// ```
    /// # #![feature(collections)]
    /// use std::collections::BitVec;
    /// let mut bv = BitVec::new();
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn new() -> BitVec {
        BitVec { storage: Vec::new(), nbits: 0 }
    }

    /// Creates a `BitVec` that holds `nbits` elements, setting each element
    /// to `bit`.
    ///
    /// # Examples
    ///
    /// ```
    /// # #![feature(collections)]
    /// use std::collections::BitVec;
    ///
    /// let mut bv = BitVec::from_elem(10, false);
    /// assert_eq!(bv.len(), 10);
    /// for x in bv.iter() {
    ///     assert_eq!(x, false);
    /// }
    /// ```
    pub fn from_elem(nbits: usize, bit: bool) -> BitVec {
        let nblocks = blocks_for_bits(nbits);
        let mut bit_vec = BitVec {
            storage: repeat(if bit { !0 } else { 0 }).take(nblocks).collect(),
            nbits: nbits
        };
        bit_vec.fix_last_block();
        bit_vec
    }

    /// Constructs a new, empty `BitVec` with the specified capacity.
    ///
    /// The bitvector will be able to hold at least `capacity` bits without
    /// reallocating. If `capacity` is 0, it will not allocate.
    ///
    /// It is important to note that this function does not specify the
    /// *length* of the returned bitvector, but only the *capacity*.
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn with_capacity(nbits: usize) -> BitVec {
        BitVec {
            storage: Vec::with_capacity(blocks_for_bits(nbits)),
            nbits: 0,
        }
    }

    /// Transforms a byte-vector into a `BitVec`. Each byte becomes eight bits,
    /// with the most significant bits of each byte coming first. Each
    /// bit becomes `true` if equal to 1 or `false` if equal to 0.
    ///
    /// # Examples
    ///
    /// ```
    /// # #![feature(collections)]
    /// use std::collections::BitVec;
    ///
    /// let bv = BitVec::from_bytes(&[0b10100000, 0b00010010]);
    /// assert!(bv.eq_vec(&[true, false, true, false,
    ///                     false, false, false, false,
    ///                     false, false, false, true,
    ///                     false, false, true, false]));
    /// ```
    pub fn from_bytes(bytes: &[u8]) -> BitVec {
        let len = bytes.len().checked_mul(u8::BITS).expect("capacity overflow");
        let mut bit_vec = BitVec::with_capacity(len);
        let complete_words = bytes.len() / 4;
        let extra_bytes = bytes.len() % 4;

        bit_vec.nbits = len;

        for i in 0..complete_words {
            bit_vec.storage.push(
                ((reverse_bits(bytes[i * 4 + 0]) as u32) << 0) |
                ((reverse_bits(bytes[i * 4 + 1]) as u32) << 8) |
                ((reverse_bits(bytes[i * 4 + 2]) as u32) << 16) |
                ((reverse_bits(bytes[i * 4 + 3]) as u32) << 24)
            );
        }

        if extra_bytes > 0 {
            let mut last_word = 0;
            for (i, &byte) in bytes[complete_words*4..].iter().enumerate() {
                last_word |= (reverse_bits(byte) as u32) << (i * 8);
            }
            bit_vec.storage.push(last_word);
        }

        bit_vec
    }

    /// Creates a `BitVec` of the specified length where the value at each index
    /// is `f(index)`.
    ///
    /// # Examples
    ///
    /// ```
    /// # #![feature(collections)]
    /// use std::collections::BitVec;
    ///
    /// let bv = BitVec::from_fn(5, |i| { i % 2 == 0 });
    /// assert!(bv.eq_vec(&[true, false, true, false, true]));
    /// ```
    pub fn from_fn<F>(len: usize, mut f: F) -> BitVec where F: FnMut(usize) -> bool {
        let mut bit_vec = BitVec::from_elem(len, false);
        for i in 0..len {
            bit_vec.set(i, f(i));
        }
        bit_vec
    }

    /// Retrieves the value at index `i`, or `None` if the index is out of bounds.
    ///
    /// # Examples
    ///
    /// ```
    /// # #![feature(collections)]
    /// use std::collections::BitVec;
    ///
    /// let bv = BitVec::from_bytes(&[0b01100000]);
    /// assert_eq!(bv.get(0), Some(false));
    /// assert_eq!(bv.get(1), Some(true));
    /// assert_eq!(bv.get(100), None);
    ///
    /// // Can also use array indexing
    /// assert_eq!(bv[1], true);
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn get(&self, i: usize) -> Option<bool> {
        if i >= self.nbits {
            return None;
        }
        let w = i / u32::BITS;
        let b = i % u32::BITS;
        self.storage.get(w).map(|&block|
            (block & (1 << b)) != 0
        )
    }

    /// Sets the value of a bit at an index `i`.
    ///
    /// # Panics
    ///
    /// Panics if `i` is out of bounds.
    ///
    /// # Examples
    ///
    /// ```
    /// # #![feature(collections)]
    /// use std::collections::BitVec;
    ///
    /// let mut bv = BitVec::from_elem(5, false);
    /// bv.set(3, true);
    /// assert_eq!(bv[3], true);
    /// ```
    #[inline]
    #[unstable(feature = "collections",
               reason = "panic semantics are likely to change in the future")]
    pub fn set(&mut self, i: usize, x: bool) {
        assert!(i < self.nbits);
        let w = i / u32::BITS;
        let b = i % u32::BITS;
        let flag = 1 << b;
        let val = if x { self.storage[w] | flag }
                  else { self.storage[w] & !flag };
        self.storage[w] = val;
    }

    /// Sets all bits to 1.
    ///
    /// # Examples
    ///
    /// ```
    /// # #![feature(collections)]
    /// use std::collections::BitVec;
    ///
    /// let before = 0b01100000;
    /// let after  = 0b11111111;
    ///
    /// let mut bv = BitVec::from_bytes(&[before]);
    /// bv.set_all();
    /// assert_eq!(bv, BitVec::from_bytes(&[after]));
    /// ```
    #[inline]
    pub fn set_all(&mut self) {
        for w in &mut self.storage { *w = !0; }
        self.fix_last_block();
    }

    /// Flips all bits.
    ///
    /// # Examples
    ///
    /// ```
    /// # #![feature(collections)]
    /// use std::collections::BitVec;
    ///
    /// let before = 0b01100000;
    /// let after  = 0b10011111;
    ///
    /// let mut bv = BitVec::from_bytes(&[before]);
    /// bv.negate();
    /// assert_eq!(bv, BitVec::from_bytes(&[after]));
    /// ```
    #[inline]
    pub fn negate(&mut self) {
        for w in &mut self.storage { *w = !*w; }
        self.fix_last_block();
    }

    /// Calculates the union of two bitvectors. This acts like the bitwise `or`
    /// function.
    ///
    /// Sets `self` to the union of `self` and `other`. Both bitvectors must be
    /// the same length. Returns `true` if `self` changed.
    ///
    /// # Panics
    ///
    /// Panics if the bitvectors are of different lengths.
    ///
    /// # Examples
    ///
    /// ```
    /// # #![feature(collections)]
    /// use std::collections::BitVec;
    ///
    /// let a   = 0b01100100;
    /// let b   = 0b01011010;
    /// let res = 0b01111110;
    ///
    /// let mut a = BitVec::from_bytes(&[a]);
    /// let b = BitVec::from_bytes(&[b]);
    ///
    /// assert!(a.union(&b));
    /// assert_eq!(a, BitVec::from_bytes(&[res]));
    /// ```
    #[inline]
    pub fn union(&mut self, other: &BitVec) -> bool {
        self.process(other, |w1, w2| w1 | w2)
    }

    /// Calculates the intersection of two bitvectors. This acts like the
    /// bitwise `and` function.
    ///
    /// Sets `self` to the intersection of `self` and `other`. Both bitvectors
    /// must be the same length. Returns `true` if `self` changed.
    ///
    /// # Panics
    ///
    /// Panics if the bitvectors are of different lengths.
    ///
    /// # Examples
    ///
    /// ```
    /// # #![feature(collections)]
    /// use std::collections::BitVec;
    ///
    /// let a   = 0b01100100;
    /// let b   = 0b01011010;
    /// let res = 0b01000000;
    ///
    /// let mut a = BitVec::from_bytes(&[a]);
    /// let b = BitVec::from_bytes(&[b]);
    ///
    /// assert!(a.intersect(&b));
    /// assert_eq!(a, BitVec::from_bytes(&[res]));
    /// ```
    #[inline]
    pub fn intersect(&mut self, other: &BitVec) -> bool {
        self.process(other, |w1, w2| w1 & w2)
    }

    /// Calculates the difference between two bitvectors.
    ///
    /// Sets each element of `self` to the value of that element minus the
    /// element of `other` at the same index. Both bitvectors must be the same
    /// length. Returns `true` if `self` changed.
    ///
    /// # Panics
    ///
    /// Panics if the bitvectors are of different length.
    ///
    /// # Examples
    ///
    /// ```
    /// # #![feature(collections)]
    /// use std::collections::BitVec;
    ///
    /// let a   = 0b01100100;
    /// let b   = 0b01011010;
    /// let a_b = 0b00100100; // a - b
    /// let b_a = 0b00011010; // b - a
    ///
    /// let mut bva = BitVec::from_bytes(&[a]);
    /// let bvb = BitVec::from_bytes(&[b]);
    ///
    /// assert!(bva.difference(&bvb));
    /// assert_eq!(bva, BitVec::from_bytes(&[a_b]));
    ///
    /// let bva = BitVec::from_bytes(&[a]);
    /// let mut bvb = BitVec::from_bytes(&[b]);
    ///
    /// assert!(bvb.difference(&bva));
    /// assert_eq!(bvb, BitVec::from_bytes(&[b_a]));
    /// ```
    #[inline]
    pub fn difference(&mut self, other: &BitVec) -> bool {
        self.process(other, |w1, w2| w1 & !w2)
    }

    /// Returns `true` if all bits are 1.
    ///
    /// # Examples
    ///
    /// ```
    /// # #![feature(collections)]
    /// use std::collections::BitVec;
    ///
    /// let mut bv = BitVec::from_elem(5, true);
    /// assert_eq!(bv.all(), true);
    ///
    /// bv.set(1, false);
    /// assert_eq!(bv.all(), false);
    /// ```
    pub fn all(&self) -> bool {
        let mut last_word = !0;
        // Check that every block but the last is all-ones...
        self.blocks().all(|elem| {
            let tmp = last_word;
            last_word = elem;
            tmp == !0
        // and then check the last one has enough ones
        }) && (last_word == mask_for_bits(self.nbits))
    }

    /// Returns an iterator over the elements of the vector in order.
    ///
    /// # Examples
    ///
    /// ```
    /// # #![feature(collections)]
    /// use std::collections::BitVec;
    ///
    /// let bv = BitVec::from_bytes(&[0b01110100, 0b10010010]);
    /// assert_eq!(bv.iter().filter(|x| *x).count(), 7);
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn iter(&self) -> Iter {
        Iter { bit_vec: self, next_idx: 0, end_idx: self.nbits }
    }

    /// Returns `true` if all bits are 0.
    ///
    /// # Examples
    ///
    /// ```
    /// # #![feature(collections)]
    /// use std::collections::BitVec;
    ///
    /// let mut bv = BitVec::from_elem(10, false);
    /// assert_eq!(bv.none(), true);
    ///
    /// bv.set(3, true);
    /// assert_eq!(bv.none(), false);
    /// ```
    pub fn none(&self) -> bool {
        self.blocks().all(|w| w == 0)
    }

    /// Returns `true` if any bit is 1.
    ///
    /// # Examples
    ///
    /// ```
    /// # #![feature(collections)]
    /// use std::collections::BitVec;
    ///
    /// let mut bv = BitVec::from_elem(10, false);
    /// assert_eq!(bv.any(), false);
    ///
    /// bv.set(3, true);
    /// assert_eq!(bv.any(), true);
    /// ```
    #[inline]
    pub fn any(&self) -> bool {
        !self.none()
    }

    /// Organises the bits into bytes, such that the first bit in the
    /// `BitVec` becomes the high-order bit of the first byte. If the
    /// size of the `BitVec` is not a multiple of eight then trailing bits
    /// will be filled-in with `false`.
    ///
    /// # Examples
    ///
    /// ```
    /// # #![feature(collections)]
    /// use std::collections::BitVec;
    ///
    /// let mut bv = BitVec::from_elem(3, true);
    /// bv.set(1, false);
    ///
    /// assert_eq!(bv.to_bytes(), [0b10100000]);
    ///
    /// let mut bv = BitVec::from_elem(9, false);
    /// bv.set(2, true);
    /// bv.set(8, true);
    ///
    /// assert_eq!(bv.to_bytes(), [0b00100000, 0b10000000]);
    /// ```
    pub fn to_bytes(&self) -> Vec<u8> {
        fn bit(bit_vec: &BitVec, byte: usize, bit: usize) -> u8 {
            let offset = byte * 8 + bit;
            if offset >= bit_vec.nbits {
                0
            } else {
                (bit_vec[offset] as u8) << (7 - bit)
            }
        }

        let len = self.nbits/8 +
                  if self.nbits % 8 == 0 { 0 } else { 1 };
        (0..len).map(|i|
            bit(self, i, 0) |
            bit(self, i, 1) |
            bit(self, i, 2) |
            bit(self, i, 3) |
            bit(self, i, 4) |
            bit(self, i, 5) |
            bit(self, i, 6) |
            bit(self, i, 7)
        ).collect()
    }

    /// Compares a `BitVec` to a slice of `bool`s.
    /// Both the `BitVec` and slice must have the same length.
    ///
    /// # Panics
    ///
    /// Panics if the `BitVec` and slice are of different length.
    ///
    /// # Examples
    ///
    /// ```
    /// # #![feature(collections)]
    /// use std::collections::BitVec;
    ///
    /// let bv = BitVec::from_bytes(&[0b10100000]);
    ///
    /// assert!(bv.eq_vec(&[true, false, true, false,
    ///                     false, false, false, false]));
    /// ```
    pub fn eq_vec(&self, v: &[bool]) -> bool {
        assert_eq!(self.nbits, v.len());
        iter::order::eq(self.iter(), v.iter().cloned())
    }

    /// Shortens a `BitVec`, dropping excess elements.
    ///
    /// If `len` is greater than the vector's current length, this has no
    /// effect.
    ///
    /// # Examples
    ///
    /// ```
    /// # #![feature(collections)]
    /// use std::collections::BitVec;
    ///
    /// let mut bv = BitVec::from_bytes(&[0b01001011]);
    /// bv.truncate(2);
    /// assert!(bv.eq_vec(&[false, true]));
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn truncate(&mut self, len: usize) {
        if len < self.len() {
            self.nbits = len;
            // This fixes (2).
            self.storage.truncate(blocks_for_bits(len));
            self.fix_last_block();
        }
    }

    /// Reserves capacity for at least `additional` more bits to be inserted in the given
    /// `BitVec`. The collection may reserve more space to avoid frequent reallocations.
    ///
    /// # Panics
    ///
    /// Panics if the new capacity overflows `usize`.
    ///
    /// # Examples
    ///
    /// ```
    /// # #![feature(collections)]
    /// use std::collections::BitVec;
    ///
    /// let mut bv = BitVec::from_elem(3, false);
    /// bv.reserve(10);
    /// assert_eq!(bv.len(), 3);
    /// assert!(bv.capacity() >= 13);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn reserve(&mut self, additional: usize) {
        let desired_cap = self.len().checked_add(additional).expect("capacity overflow");
        let storage_len = self.storage.len();
        if desired_cap > self.capacity() {
            self.storage.reserve(blocks_for_bits(desired_cap) - storage_len);
        }
    }

    /// Reserves the minimum capacity for exactly `additional` more bits to be inserted in the
    /// given `BitVec`. Does nothing if the capacity is already sufficient.
    ///
    /// Note that the allocator may give the collection more space than it requests. Therefore
    /// capacity can not be relied upon to be precisely minimal. Prefer `reserve` if future
    /// insertions are expected.
    ///
    /// # Panics
    ///
    /// Panics if the new capacity overflows `usize`.
    ///
    /// # Examples
    ///
    /// ```
    /// # #![feature(collections)]
    /// use std::collections::BitVec;
    ///
    /// let mut bv = BitVec::from_elem(3, false);
    /// bv.reserve(10);
    /// assert_eq!(bv.len(), 3);
    /// assert!(bv.capacity() >= 13);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn reserve_exact(&mut self, additional: usize) {
        let desired_cap = self.len().checked_add(additional).expect("capacity overflow");
        let storage_len = self.storage.len();
        if desired_cap > self.capacity() {
            self.storage.reserve_exact(blocks_for_bits(desired_cap) - storage_len);
        }
    }

    /// Returns the capacity in bits for this bit vector. Inserting any
    /// element less than this amount will not trigger a resizing.
    ///
    /// # Examples
    ///
    /// ```
    /// # #![feature(collections)]
    /// use std::collections::BitVec;
    ///
    /// let mut bv = BitVec::new();
    /// bv.reserve(10);
    /// assert!(bv.capacity() >= 10);
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn capacity(&self) -> usize {
        self.storage.capacity().checked_mul(u32::BITS).unwrap_or(usize::MAX)
    }

    /// Grows the `BitVec` in-place, adding `n` copies of `value` to the `BitVec`.
    ///
    /// # Panics
    ///
    /// Panics if the new len overflows a `usize`.
    ///
    /// # Examples
    ///
    /// ```
    /// # #![feature(collections)]
    /// use std::collections::BitVec;
    ///
    /// let mut bv = BitVec::from_bytes(&[0b01001011]);
    /// bv.grow(2, true);
    /// assert_eq!(bv.len(), 10);
    /// assert_eq!(bv.to_bytes(), [0b01001011, 0b11000000]);
    /// ```
    pub fn grow(&mut self, n: usize, value: bool) {
        // Note: we just bulk set all the bits in the last word in this fn in multiple places
        // which is technically wrong if not all of these bits are to be used. However, at the end
        // of this fn we call `fix_last_block` at the end of this fn, which should fix this.

        let new_nbits = self.nbits.checked_add(n).expect("capacity overflow");
        let new_nblocks = blocks_for_bits(new_nbits);
        let full_value = if value { !0 } else { 0 };

        // Correct the old tail word, setting or clearing formerly unused bits
        let num_cur_blocks = blocks_for_bits(self.nbits);
        if self.nbits % u32::BITS > 0 {
            let mask = mask_for_bits(self.nbits);
            if value {
                self.storage[num_cur_blocks - 1] |= !mask;
            } else {
                // Extra bits are already zero by invariant.
            }
        }

        // Fill in words after the old tail word
        let stop_idx = cmp::min(self.storage.len(), new_nblocks);
        for idx in num_cur_blocks..stop_idx {
            self.storage[idx] = full_value;
        }

        // Allocate new words, if needed
        if new_nblocks > self.storage.len() {
            let to_add = new_nblocks - self.storage.len();
            self.storage.extend(repeat(full_value).take(to_add));
        }

        // Adjust internal bit count
        self.nbits = new_nbits;

        self.fix_last_block();
    }

    /// Removes the last bit from the BitVec, and returns it. Returns None if the BitVec is empty.
    ///
    /// # Examples
    ///
    /// ```
    /// # #![feature(collections)]
    /// use std::collections::BitVec;
    ///
    /// let mut bv = BitVec::from_bytes(&[0b01001001]);
    /// assert_eq!(bv.pop(), Some(true));
    /// assert_eq!(bv.pop(), Some(false));
    /// assert_eq!(bv.len(), 6);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn pop(&mut self) -> Option<bool> {
        if self.is_empty() {
            None
        } else {
            let i = self.nbits - 1;
            let ret = self[i];
            // (3)
            self.set(i, false);
            self.nbits = i;
            if self.nbits % u32::BITS == 0 {
                // (2)
                self.storage.pop();
            }
            Some(ret)
        }
    }

    /// Pushes a `bool` onto the end.
    ///
    /// # Examples
    ///
    /// ```
    /// # #![feature(collections)]
    /// use std::collections::BitVec;
    ///
    /// let mut bv = BitVec::new();
    /// bv.push(true);
    /// bv.push(false);
    /// assert!(bv.eq_vec(&[true, false]));
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn push(&mut self, elem: bool) {
        if self.nbits % u32::BITS == 0 {
            self.storage.push(0);
        }
        let insert_pos = self.nbits;
        self.nbits = self.nbits.checked_add(1).expect("Capacity overflow");
        self.set(insert_pos, elem);
    }

    /// Returns the total number of bits in this vector
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn len(&self) -> usize { self.nbits }

    /// Returns true if there are no bits in this vector
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn is_empty(&self) -> bool { self.len() == 0 }

    /// Clears all bits in this vector.
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn clear(&mut self) {
        for w in &mut self.storage { *w = 0; }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl Default for BitVec {
    /// Creates a new `BitVec` using `BitVec::new`
    #[inline]
    fn default() -> BitVec { BitVec::new() }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl FromIterator<bool> for BitVec {
    fn from_iter<I: IntoIterator<Item=bool>>(iter: I) -> BitVec {
        let mut ret = BitVec::new();
        ret.extend(iter);
        ret
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl Extend<bool> for BitVec {
    #[inline]
    fn extend<I: IntoIterator<Item=bool>>(&mut self, iterable: I) {
        let iterator = iterable.into_iter();
        let (min, _) = iterator.size_hint();
        self.reserve(min);
        for element in iterator {
            self.push(element)
        }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl Clone for BitVec {
    #[inline]
    fn clone(&self) -> BitVec {
        BitVec { storage: self.storage.clone(), nbits: self.nbits }
    }

    #[inline]
    fn clone_from(&mut self, source: &BitVec) {
        self.nbits = source.nbits;
        self.storage.clone_from(&source.storage);
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl PartialOrd for BitVec {
    #[inline]
    fn partial_cmp(&self, other: &BitVec) -> Option<Ordering> {
        iter::order::partial_cmp(self.iter(), other.iter())
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl Ord for BitVec {
    #[inline]
    fn cmp(&self, other: &BitVec) -> Ordering {
        iter::order::cmp(self.iter(), other.iter())
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl fmt::Debug for BitVec {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        for bit in self {
            try!(write!(fmt, "{}", if bit { 1 } else { 0 }));
        }
        Ok(())
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl hash::Hash for BitVec {
    fn hash<H: hash::Hasher>(&self, state: &mut H) {
        self.nbits.hash(state);
        for elem in self.blocks() {
            elem.hash(state);
        }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl cmp::PartialEq for BitVec {
    #[inline]
    fn eq(&self, other: &BitVec) -> bool {
        if self.nbits != other.nbits {
            return false;
        }
        self.blocks().zip(other.blocks()).all(|(w1, w2)| w1 == w2)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl cmp::Eq for BitVec {}

/// An iterator for `BitVec`.
#[stable(feature = "rust1", since = "1.0.0")]
#[derive(Clone)]
pub struct Iter<'a> {
    bit_vec: &'a BitVec,
    next_idx: usize,
    end_idx: usize,
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a> Iterator for Iter<'a> {
    type Item = bool;

    #[inline]
    fn next(&mut self) -> Option<bool> {
        if self.next_idx != self.end_idx {
            let idx = self.next_idx;
            self.next_idx += 1;
            Some(self.bit_vec[idx])
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let rem = self.end_idx - self.next_idx;
        (rem, Some(rem))
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a> DoubleEndedIterator for Iter<'a> {
    #[inline]
    fn next_back(&mut self) -> Option<bool> {
        if self.next_idx != self.end_idx {
            self.end_idx -= 1;
            Some(self.bit_vec[self.end_idx])
        } else {
            None
        }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a> ExactSizeIterator for Iter<'a> {}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a> RandomAccessIterator for Iter<'a> {
    #[inline]
    fn indexable(&self) -> usize {
        self.end_idx - self.next_idx
    }

    #[inline]
    fn idx(&mut self, index: usize) -> Option<bool> {
        if index >= self.indexable() {
            None
        } else {
            Some(self.bit_vec[index])
        }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a> IntoIterator for &'a BitVec {
    type Item = bool;
    type IntoIter = Iter<'a>;

    fn into_iter(self) -> Iter<'a> {
        self.iter()
    }
}

/// An implementation of a set using a bit vector as an underlying
/// representation for holding unsigned numerical elements.
///
/// It should also be noted that the amount of storage necessary for holding a
/// set of objects is proportional to the maximum of the objects when viewed
/// as a `usize`.
///
/// # Examples
///
/// ```
/// # #![feature(collections)]
/// use std::collections::{BitSet, BitVec};
///
/// // It's a regular set
/// let mut s = BitSet::new();
/// s.insert(0);
/// s.insert(3);
/// s.insert(7);
///
/// s.remove(&7);
///
/// if !s.contains(&7) {
///     println!("There is no 7");
/// }
///
/// // Can initialize from a `BitVec`
/// let other = BitSet::from_bit_vec(BitVec::from_bytes(&[0b11010000]));
///
/// s.union_with(&other);
///
/// // Print 0, 1, 3 in some order
/// for x in s.iter() {
///     println!("{}", x);
/// }
///
/// // Can convert back to a `BitVec`
/// let bv: BitVec = s.into_bit_vec();
/// assert!(bv[3]);
/// ```
#[derive(Clone)]
#[unstable(feature = "collections",
           reason = "RFC 509")]
pub struct BitSet {
    bit_vec: BitVec,
}

#[stable(feature = "rust1", since = "1.0.0")]
impl Default for BitSet {
    /// Creates a new `BitSet` using `BitSet::new`
    #[inline]
    fn default() -> BitSet { BitSet::new() }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl FromIterator<usize> for BitSet {
    fn from_iter<I: IntoIterator<Item=usize>>(iter: I) -> BitSet {
        let mut ret = BitSet::new();
        ret.extend(iter);
        ret
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl Extend<usize> for BitSet {
    #[inline]
    fn extend<I: IntoIterator<Item=usize>>(&mut self, iter: I) {
        for i in iter {
            self.insert(i);
        }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl PartialOrd for BitSet {
    #[inline]
    fn partial_cmp(&self, other: &BitSet) -> Option<Ordering> {
        let (a_iter, b_iter) = match_words(self.get_ref(), other.get_ref());
        iter::order::partial_cmp(a_iter, b_iter)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl Ord for BitSet {
    #[inline]
    fn cmp(&self, other: &BitSet) -> Ordering {
        let (a_iter, b_iter) = match_words(self.get_ref(), other.get_ref());
        iter::order::cmp(a_iter, b_iter)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl cmp::PartialEq for BitSet {
    #[inline]
    fn eq(&self, other: &BitSet) -> bool {
        let (a_iter, b_iter) = match_words(self.get_ref(), other.get_ref());
        iter::order::eq(a_iter, b_iter)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl cmp::Eq for BitSet {}

impl BitSet {
    /// Creates a new empty `BitSet`.
    ///
    /// # Examples
    ///
    /// ```
    /// # #![feature(collections)]
    /// use std::collections::BitSet;
    ///
    /// let mut s = BitSet::new();
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn new() -> BitSet {
        BitSet { bit_vec: BitVec::new() }
    }

    /// Creates a new `BitSet` with initially no contents, able to
    /// hold `nbits` elements without resizing.
    ///
    /// # Examples
    ///
    /// ```
    /// # #![feature(collections)]
    /// use std::collections::BitSet;
    ///
    /// let mut s = BitSet::with_capacity(100);
    /// assert!(s.capacity() >= 100);
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn with_capacity(nbits: usize) -> BitSet {
        let bit_vec = BitVec::from_elem(nbits, false);
        BitSet::from_bit_vec(bit_vec)
    }

    /// Creates a new `BitSet` from the given bit vector.
    ///
    /// # Examples
    ///
    /// ```
    /// # #![feature(collections)]
    /// use std::collections::{BitVec, BitSet};
    ///
    /// let bv = BitVec::from_bytes(&[0b01100000]);
    /// let s = BitSet::from_bit_vec(bv);
    ///
    /// // Print 1, 2 in arbitrary order
    /// for x in s.iter() {
    ///     println!("{}", x);
    /// }
    /// ```
    #[inline]
    pub fn from_bit_vec(bit_vec: BitVec) -> BitSet {
        BitSet { bit_vec: bit_vec }
    }

    /// Returns the capacity in bits for this bit vector. Inserting any
    /// element less than this amount will not trigger a resizing.
    ///
    /// # Examples
    ///
    /// ```
    /// # #![feature(collections)]
    /// use std::collections::BitSet;
    ///
    /// let mut s = BitSet::with_capacity(100);
    /// assert!(s.capacity() >= 100);
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn capacity(&self) -> usize {
        self.bit_vec.capacity()
    }

    /// Reserves capacity for the given `BitSet` to contain `len` distinct elements. In the case
    /// of `BitSet` this means reallocations will not occur as long as all inserted elements
    /// are less than `len`.
    ///
    /// The collection may reserve more space to avoid frequent reallocations.
    ///
    ///
    /// # Examples
    ///
    /// ```
    /// # #![feature(collections)]
    /// use std::collections::BitSet;
    ///
    /// let mut s = BitSet::new();
    /// s.reserve_len(10);
    /// assert!(s.capacity() >= 10);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn reserve_len(&mut self, len: usize) {
        let cur_len = self.bit_vec.len();
        if len >= cur_len {
            self.bit_vec.reserve(len - cur_len);
        }
    }

    /// Reserves the minimum capacity for the given `BitSet` to contain `len` distinct elements.
    /// In the case of `BitSet` this means reallocations will not occur as long as all inserted
    /// elements are less than `len`.
    ///
    /// Note that the allocator may give the collection more space than it requests. Therefore
    /// capacity can not be relied upon to be precisely minimal. Prefer `reserve_len` if future
    /// insertions are expected.
    ///
    ///
    /// # Examples
    ///
    /// ```
    /// # #![feature(collections)]
    /// use std::collections::BitSet;
    ///
    /// let mut s = BitSet::new();
    /// s.reserve_len_exact(10);
    /// assert!(s.capacity() >= 10);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn reserve_len_exact(&mut self, len: usize) {
        let cur_len = self.bit_vec.len();
        if len >= cur_len {
            self.bit_vec.reserve_exact(len - cur_len);
        }
    }


    /// Consumes this set to return the underlying bit vector.
    ///
    /// # Examples
    ///
    /// ```
    /// # #![feature(collections)]
    /// use std::collections::BitSet;
    ///
    /// let mut s = BitSet::new();
    /// s.insert(0);
    /// s.insert(3);
    ///
    /// let bv = s.into_bit_vec();
    /// assert!(bv[0]);
    /// assert!(bv[3]);
    /// ```
    #[inline]
    pub fn into_bit_vec(self) -> BitVec {
        self.bit_vec
    }

    /// Returns a reference to the underlying bit vector.
    ///
    /// # Examples
    ///
    /// ```
    /// # #![feature(collections)]
    /// use std::collections::BitSet;
    ///
    /// let mut s = BitSet::new();
    /// s.insert(0);
    ///
    /// let bv = s.get_ref();
    /// assert_eq!(bv[0], true);
    /// ```
    #[inline]
    pub fn get_ref(&self) -> &BitVec {
        &self.bit_vec
    }

    #[inline]
    fn other_op<F>(&mut self, other: &BitSet, mut f: F) where F: FnMut(u32, u32) -> u32 {
        // Unwrap BitVecs
        let self_bit_vec = &mut self.bit_vec;
        let other_bit_vec = &other.bit_vec;

        let self_len = self_bit_vec.len();
        let other_len = other_bit_vec.len();

        // Expand the vector if necessary
        if self_len < other_len {
            self_bit_vec.grow(other_len - self_len, false);
        }

        // virtually pad other with 0's for equal lengths
        let other_words = {
            let (_, result) = match_words(self_bit_vec, other_bit_vec);
            result
        };

        // Apply values found in other
        for (i, w) in other_words {
            let old = self_bit_vec.storage[i];
            let new = f(old, w);
            self_bit_vec.storage[i] = new;
        }
    }

    /// Truncates the underlying vector to the least length required.
    ///
    /// # Examples
    ///
    /// ```
    /// # #![feature(collections)]
    /// use std::collections::BitSet;
    ///
    /// let mut s = BitSet::new();
    /// s.insert(32183231);
    /// s.remove(&32183231);
    ///
    /// // Internal storage will probably be bigger than necessary
    /// println!("old capacity: {}", s.capacity());
    ///
    /// // Now should be smaller
    /// s.shrink_to_fit();
    /// println!("new capacity: {}", s.capacity());
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn shrink_to_fit(&mut self) {
        let bit_vec = &mut self.bit_vec;
        // Obtain original length
        let old_len = bit_vec.storage.len();
        // Obtain coarse trailing zero length
        let n = bit_vec.storage.iter().rev().take_while(|&&n| n == 0).count();
        // Truncate
        let trunc_len = cmp::max(old_len - n, 1);
        bit_vec.storage.truncate(trunc_len);
        bit_vec.nbits = trunc_len * u32::BITS;
    }

    /// Iterator over each u32 stored in the `BitSet`.
    ///
    /// # Examples
    ///
    /// ```
    /// # #![feature(collections)]
    /// use std::collections::{BitVec, BitSet};
    ///
    /// let s = BitSet::from_bit_vec(BitVec::from_bytes(&[0b01001010]));
    ///
    /// // Print 1, 4, 6 in arbitrary order
    /// for x in s.iter() {
    ///     println!("{}", x);
    /// }
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn iter(&self) -> bit_set::Iter {
        SetIter {set: self, next_idx: 0}
    }

    /// Iterator over each u32 stored in `self` union `other`.
    /// See [union_with](#method.union_with) for an efficient in-place version.
    ///
    /// # Examples
    ///
    /// ```
    /// # #![feature(collections)]
    /// use std::collections::{BitVec, BitSet};
    ///
    /// let a = BitSet::from_bit_vec(BitVec::from_bytes(&[0b01101000]));
    /// let b = BitSet::from_bit_vec(BitVec::from_bytes(&[0b10100000]));
    ///
    /// // Print 0, 1, 2, 4 in arbitrary order
    /// for x in a.union(&b) {
    ///     println!("{}", x);
    /// }
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn union<'a>(&'a self, other: &'a BitSet) -> Union<'a> {
        fn or(w1: u32, w2: u32) -> u32 { w1 | w2 }

        Union(TwoBitPositions {
            set: self,
            other: other,
            merge: or,
            current_word: 0,
            next_idx: 0
        })
    }

    /// Iterator over each usize stored in `self` intersect `other`.
    /// See [intersect_with](#method.intersect_with) for an efficient in-place version.
    ///
    /// # Examples
    ///
    /// ```
    /// # #![feature(collections)]
    /// use std::collections::{BitVec, BitSet};
    ///
    /// let a = BitSet::from_bit_vec(BitVec::from_bytes(&[0b01101000]));
    /// let b = BitSet::from_bit_vec(BitVec::from_bytes(&[0b10100000]));
    ///
    /// // Print 2
    /// for x in a.intersection(&b) {
    ///     println!("{}", x);
    /// }
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn intersection<'a>(&'a self, other: &'a BitSet) -> Intersection<'a> {
        fn bitand(w1: u32, w2: u32) -> u32 { w1 & w2 }
        let min = cmp::min(self.bit_vec.len(), other.bit_vec.len());
        Intersection(TwoBitPositions {
            set: self,
            other: other,
            merge: bitand,
            current_word: 0,
            next_idx: 0
        }.take(min))
    }

    /// Iterator over each usize stored in the `self` setminus `other`.
    /// See [difference_with](#method.difference_with) for an efficient in-place version.
    ///
    /// # Examples
    ///
    /// ```
    /// # #![feature(collections)]
    /// use std::collections::{BitSet, BitVec};
    ///
    /// let a = BitSet::from_bit_vec(BitVec::from_bytes(&[0b01101000]));
    /// let b = BitSet::from_bit_vec(BitVec::from_bytes(&[0b10100000]));
    ///
    /// // Print 1, 4 in arbitrary order
    /// for x in a.difference(&b) {
    ///     println!("{}", x);
    /// }
    ///
    /// // Note that difference is not symmetric,
    /// // and `b - a` means something else.
    /// // This prints 0
    /// for x in b.difference(&a) {
    ///     println!("{}", x);
    /// }
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn difference<'a>(&'a self, other: &'a BitSet) -> Difference<'a> {
        fn diff(w1: u32, w2: u32) -> u32 { w1 & !w2 }

        Difference(TwoBitPositions {
            set: self,
            other: other,
            merge: diff,
            current_word: 0,
            next_idx: 0
        })
    }

    /// Iterator over each u32 stored in the symmetric difference of `self` and `other`.
    /// See [symmetric_difference_with](#method.symmetric_difference_with) for
    /// an efficient in-place version.
    ///
    /// # Examples
    ///
    /// ```
    /// # #![feature(collections)]
    /// use std::collections::{BitSet, BitVec};
    ///
    /// let a = BitSet::from_bit_vec(BitVec::from_bytes(&[0b01101000]));
    /// let b = BitSet::from_bit_vec(BitVec::from_bytes(&[0b10100000]));
    ///
    /// // Print 0, 1, 4 in arbitrary order
    /// for x in a.symmetric_difference(&b) {
    ///     println!("{}", x);
    /// }
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn symmetric_difference<'a>(&'a self, other: &'a BitSet) -> SymmetricDifference<'a> {
        fn bitxor(w1: u32, w2: u32) -> u32 { w1 ^ w2 }

        SymmetricDifference(TwoBitPositions {
            set: self,
            other: other,
            merge: bitxor,
            current_word: 0,
            next_idx: 0
        })
    }

    /// Unions in-place with the specified other bit vector.
    ///
    /// # Examples
    ///
    /// ```
    /// # #![feature(collections)]
    /// use std::collections::{BitSet, BitVec};
    ///
    /// let a   = 0b01101000;
    /// let b   = 0b10100000;
    /// let res = 0b11101000;
    ///
    /// let mut a = BitSet::from_bit_vec(BitVec::from_bytes(&[a]));
    /// let b = BitSet::from_bit_vec(BitVec::from_bytes(&[b]));
    /// let res = BitSet::from_bit_vec(BitVec::from_bytes(&[res]));
    ///
    /// a.union_with(&b);
    /// assert_eq!(a, res);
    /// ```
    #[inline]
    pub fn union_with(&mut self, other: &BitSet) {
        self.other_op(other, |w1, w2| w1 | w2);
    }

    /// Intersects in-place with the specified other bit vector.
    ///
    /// # Examples
    ///
    /// ```
    /// # #![feature(collections)]
    /// use std::collections::{BitSet, BitVec};
    ///
    /// let a   = 0b01101000;
    /// let b   = 0b10100000;
    /// let res = 0b00100000;
    ///
    /// let mut a = BitSet::from_bit_vec(BitVec::from_bytes(&[a]));
    /// let b = BitSet::from_bit_vec(BitVec::from_bytes(&[b]));
    /// let res = BitSet::from_bit_vec(BitVec::from_bytes(&[res]));
    ///
    /// a.intersect_with(&b);
    /// assert_eq!(a, res);
    /// ```
    #[inline]
    pub fn intersect_with(&mut self, other: &BitSet) {
        self.other_op(other, |w1, w2| w1 & w2);
    }

    /// Makes this bit vector the difference with the specified other bit vector
    /// in-place.
    ///
    /// # Examples
    ///
    /// ```
    /// # #![feature(collections)]
    /// use std::collections::{BitSet, BitVec};
    ///
    /// let a   = 0b01101000;
    /// let b   = 0b10100000;
    /// let a_b = 0b01001000; // a - b
    /// let b_a = 0b10000000; // b - a
    ///
    /// let mut bva = BitSet::from_bit_vec(BitVec::from_bytes(&[a]));
    /// let bvb = BitSet::from_bit_vec(BitVec::from_bytes(&[b]));
    /// let bva_b = BitSet::from_bit_vec(BitVec::from_bytes(&[a_b]));
    /// let bvb_a = BitSet::from_bit_vec(BitVec::from_bytes(&[b_a]));
    ///
    /// bva.difference_with(&bvb);
    /// assert_eq!(bva, bva_b);
    ///
    /// let bva = BitSet::from_bit_vec(BitVec::from_bytes(&[a]));
    /// let mut bvb = BitSet::from_bit_vec(BitVec::from_bytes(&[b]));
    ///
    /// bvb.difference_with(&bva);
    /// assert_eq!(bvb, bvb_a);
    /// ```
    #[inline]
    pub fn difference_with(&mut self, other: &BitSet) {
        self.other_op(other, |w1, w2| w1 & !w2);
    }

    /// Makes this bit vector the symmetric difference with the specified other
    /// bit vector in-place.
    ///
    /// # Examples
    ///
    /// ```
    /// # #![feature(collections)]
    /// use std::collections::{BitSet, BitVec};
    ///
    /// let a   = 0b01101000;
    /// let b   = 0b10100000;
    /// let res = 0b11001000;
    ///
    /// let mut a = BitSet::from_bit_vec(BitVec::from_bytes(&[a]));
    /// let b = BitSet::from_bit_vec(BitVec::from_bytes(&[b]));
    /// let res = BitSet::from_bit_vec(BitVec::from_bytes(&[res]));
    ///
    /// a.symmetric_difference_with(&b);
    /// assert_eq!(a, res);
    /// ```
    #[inline]
    pub fn symmetric_difference_with(&mut self, other: &BitSet) {
        self.other_op(other, |w1, w2| w1 ^ w2);
    }

    /// Returns the number of set bits in this set.
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn len(&self) -> usize  {
        self.bit_vec.blocks().fold(0, |acc, n| acc + n.count_ones() as usize)
    }

    /// Returns whether there are no bits set in this set
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn is_empty(&self) -> bool {
        self.bit_vec.none()
    }

    /// Clears all bits in this set
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn clear(&mut self) {
        self.bit_vec.clear();
    }

    /// Returns `true` if this set contains the specified integer.
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn contains(&self, value: &usize) -> bool {
        let bit_vec = &self.bit_vec;
        *value < bit_vec.nbits && bit_vec[*value]
    }

    /// Returns `true` if the set has no elements in common with `other`.
    /// This is equivalent to checking for an empty intersection.
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn is_disjoint(&self, other: &BitSet) -> bool {
        self.intersection(other).next().is_none()
    }

    /// Returns `true` if the set is a subset of another.
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn is_subset(&self, other: &BitSet) -> bool {
        let self_bit_vec = &self.bit_vec;
        let other_bit_vec = &other.bit_vec;
        let other_blocks = blocks_for_bits(other_bit_vec.len());

        // Check that `self` intersect `other` is self
        self_bit_vec.blocks().zip(other_bit_vec.blocks()).all(|(w1, w2)| w1 & w2 == w1) &&
        // Make sure if `self` has any more blocks than `other`, they're all 0
        self_bit_vec.blocks().skip(other_blocks).all(|w| w == 0)
    }

    /// Returns `true` if the set is a superset of another.
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn is_superset(&self, other: &BitSet) -> bool {
        other.is_subset(self)
    }

    /// Adds a value to the set. Returns `true` if the value was not already
    /// present in the set.
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn insert(&mut self, value: usize) -> bool {
        if self.contains(&value) {
            return false;
        }

        // Ensure we have enough space to hold the new element
        let len = self.bit_vec.len();
        if value >= len {
            self.bit_vec.grow(value - len + 1, false)
        }

        self.bit_vec.set(value, true);
        return true;
    }

    /// Removes a value from the set. Returns `true` if the value was
    /// present in the set.
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn remove(&mut self, value: &usize) -> bool {
        if !self.contains(value) {
            return false;
        }

        self.bit_vec.set(*value, false);

        return true;
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl fmt::Debug for BitSet {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        try!(write!(fmt, "{{"));
        let mut first = true;
        for n in self {
            if !first {
                try!(write!(fmt, ", "));
            }
            try!(write!(fmt, "{:?}", n));
            first = false;
        }
        write!(fmt, "}}")
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl hash::Hash for BitSet {
    fn hash<H: hash::Hasher>(&self, state: &mut H) {
        for pos in self {
            pos.hash(state);
        }
    }
}

/// An iterator for `BitSet`.
#[derive(Clone)]
#[stable(feature = "rust1", since = "1.0.0")]
pub struct SetIter<'a> {
    set: &'a BitSet,
    next_idx: usize
}

/// An iterator combining two `BitSet` iterators.
#[derive(Clone)]
struct TwoBitPositions<'a> {
    set: &'a BitSet,
    other: &'a BitSet,
    merge: fn(u32, u32) -> u32,
    current_word: u32,
    next_idx: usize
}

#[derive(Clone)]
#[stable(feature = "rust1", since = "1.0.0")]
pub struct Union<'a>(TwoBitPositions<'a>);
#[derive(Clone)]
#[stable(feature = "rust1", since = "1.0.0")]
pub struct Intersection<'a>(Take<TwoBitPositions<'a>>);
#[derive(Clone)]
#[stable(feature = "rust1", since = "1.0.0")]
pub struct Difference<'a>(TwoBitPositions<'a>);
#[derive(Clone)]
#[stable(feature = "rust1", since = "1.0.0")]
pub struct SymmetricDifference<'a>(TwoBitPositions<'a>);

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a> Iterator for SetIter<'a> {
    type Item = usize;

    fn next(&mut self) -> Option<usize> {
        while self.next_idx < self.set.bit_vec.len() {
            let idx = self.next_idx;
            self.next_idx += 1;

            if self.set.contains(&idx) {
                return Some(idx);
            }
        }

        return None;
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, Some(self.set.bit_vec.len() - self.next_idx))
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a> Iterator for TwoBitPositions<'a> {
    type Item = usize;

    fn next(&mut self) -> Option<usize> {
        while self.next_idx < self.set.bit_vec.len() ||
              self.next_idx < self.other.bit_vec.len() {
            let bit_idx = self.next_idx % u32::BITS;
            if bit_idx == 0 {
                let s_bit_vec = &self.set.bit_vec;
                let o_bit_vec = &self.other.bit_vec;
                // Merging the two words is a bit of an awkward dance since
                // one BitVec might be longer than the other
                let word_idx = self.next_idx / u32::BITS;
                let w1 = if word_idx < s_bit_vec.storage.len() {
                             s_bit_vec.storage[word_idx]
                         } else { 0 };
                let w2 = if word_idx < o_bit_vec.storage.len() {
                             o_bit_vec.storage[word_idx]
                         } else { 0 };
                self.current_word = (self.merge)(w1, w2);
            }

            self.next_idx += 1;
            if self.current_word & (1 << bit_idx) != 0 {
                return Some(self.next_idx - 1);
            }
        }
        return None;
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let cap = cmp::max(self.set.bit_vec.len(), self.other.bit_vec.len());
        (0, Some(cap - self.next_idx))
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a> Iterator for Union<'a> {
    type Item = usize;

    #[inline] fn next(&mut self) -> Option<usize> { self.0.next() }
    #[inline] fn size_hint(&self) -> (usize, Option<usize>) { self.0.size_hint() }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a> Iterator for Intersection<'a> {
    type Item = usize;

    #[inline] fn next(&mut self) -> Option<usize> { self.0.next() }
    #[inline] fn size_hint(&self) -> (usize, Option<usize>) { self.0.size_hint() }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a> Iterator for Difference<'a> {
    type Item = usize;

    #[inline] fn next(&mut self) -> Option<usize> { self.0.next() }
    #[inline] fn size_hint(&self) -> (usize, Option<usize>) { self.0.size_hint() }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a> Iterator for SymmetricDifference<'a> {
    type Item = usize;

    #[inline] fn next(&mut self) -> Option<usize> { self.0.next() }
    #[inline] fn size_hint(&self) -> (usize, Option<usize>) { self.0.size_hint() }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a> IntoIterator for &'a BitSet {
    type Item = usize;
    type IntoIter = SetIter<'a>;

    fn into_iter(self) -> SetIter<'a> {
        self.iter()
    }
}
