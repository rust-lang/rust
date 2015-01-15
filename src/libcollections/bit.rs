// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// FIXME(Gankro): Bitv and BitvSet are very tightly coupled. Ideally (for
// maintenance), they should be in separate files/modules, with BitvSet only
// using Bitv's public API. This will be hard for performance though, because
// `Bitv` will not want to leak its internal representation while its internal
// representation as `u32`s must be assumed for best performance.

// FIXME(tbu-): `Bitv`'s methods shouldn't be `union`, `intersection`, but
// rather `or` and `and`.

// (1) Be careful, most things can overflow here because the amount of bits in
//     memory can overflow `uint`.
// (2) Make sure that the underlying vector has no excess length:
//     E. g. `nbits == 16`, `storage.len() == 2` would be excess length,
//     because the last word isn't used at all. This is important because some
//     methods rely on it (for *CORRECTNESS*).
// (3) Make sure that the unused bits in the last word are zeroed out, again
//     other methods rely on it for *CORRECTNESS*.
// (4) `BitvSet` is tightly coupled with `Bitv`, so any changes you make in
// `Bitv` will need to be reflected in `BitvSet`.

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
//! use std::collections::{BitvSet, Bitv};
//! use std::num::Float;
//! use std::iter;
//!
//! let max_prime = 10000;
//!
//! // Store the primes as a BitvSet
//! let primes = {
//!     // Assume all numbers are prime to begin, and then we
//!     // cross off non-primes progressively
//!     let mut bv = Bitv::from_elem(max_prime, true);
//!
//!     // Neither 0 nor 1 are prime
//!     bv.set(0, false);
//!     bv.set(1, false);
//!
//!     for i in iter::range_inclusive(2, (max_prime as f64).sqrt() as uint) {
//!         // if i is a prime
//!         if bv[i] {
//!             // Mark all multiples of i as non-prime (any multiples below i * i
//!             // will have been marked as non-prime previously)
//!             for j in iter::range_step(i * i, max_prime, i) { bv.set(j, false) }
//!         }
//!     }
//!     BitvSet::from_bitv(bv)
//! };
//!
//! // Simple primality tests below our max bound
//! let print_primes = 20;
//! print!("The primes below {} are: ", print_primes);
//! for x in range(0, print_primes) {
//!     if primes.contains(&x) {
//!         print!("{} ", x);
//!     }
//! }
//! println!("");
//!
//! // We can manipulate the internal Bitv
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
use core::iter::{self, FromIterator};
use core::num::Int;
use core::ops::Index;
use core::slice;
use core::{u8, u32, uint};
use bitv_set; //so meta

use Vec;

type Blocks<'a> = Cloned<slice::Iter<'a, u32>>;
type MutBlocks<'a> = slice::IterMut<'a, u32>;
type MatchWords<'a> = Chain<Enumerate<Blocks<'a>>, Skip<Take<Enumerate<Repeat<u32>>>>>;

fn reverse_bits(byte: u8) -> u8 {
    let mut result = 0;
    for i in range(0, u8::BITS) {
        result |= ((byte >> i) & 1) << (u8::BITS - 1 - i);
    }
    result
}

// Take two BitV's, and return iterators of their words, where the shorter one
// has been padded with 0's
fn match_words <'a,'b>(a: &'a Bitv, b: &'b Bitv) -> (MatchWords<'a>, MatchWords<'b>) {
    let a_len = a.storage.len();
    let b_len = b.storage.len();

    // have to uselessly pretend to pad the longer one for type matching
    if a_len < b_len {
        (a.blocks().enumerate().chain(iter::repeat(0u32).enumerate().take(b_len).skip(a_len)),
         b.blocks().enumerate().chain(iter::repeat(0u32).enumerate().take(0).skip(0)))
    } else {
        (a.blocks().enumerate().chain(iter::repeat(0u32).enumerate().take(0).skip(0)),
         b.blocks().enumerate().chain(iter::repeat(0u32).enumerate().take(a_len).skip(b_len)))
    }
}

static TRUE: bool = true;
static FALSE: bool = false;

/// The bitvector type.
///
/// # Examples
///
/// ```rust
/// use std::collections::Bitv;
///
/// let mut bv = Bitv::from_elem(10, false);
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
#[stable]
pub struct Bitv {
    /// Internal representation of the bit vector
    storage: Vec<u32>,
    /// The number of valid bits in the internal representation
    nbits: uint
}

// FIXME(Gankro): NopeNopeNopeNopeNope (wait for IndexGet to be a thing)
impl Index<uint> for Bitv {
    type Output = bool;

    #[inline]
    fn index(&self, i: &uint) -> &bool {
        if self.get(*i).expect("index out of bounds") {
            &TRUE
        } else {
            &FALSE
        }
    }
}

/// Computes how many blocks are needed to store that many bits
fn blocks_for_bits(bits: uint) -> uint {
    // If we want 17 bits, dividing by 32 will produce 0. So we add 1 to make sure we
    // reserve enough. But if we want exactly a multiple of 32, this will actually allocate
    // one too many. So we need to check if that's the case. We can do that by computing if
    // bitwise AND by `32 - 1` is 0. But LLVM should be able to optimize the semantically
    // superior modulo operator on a power of two to this.
    //
    // Note that we can technically avoid this branch with the expression
    // `(nbits + u32::BITS - 1) / 32::BITS`, but if nbits is almost uint::MAX this will overflow.
    if bits % u32::BITS == 0 {
        bits / u32::BITS
    } else {
        bits / u32::BITS + 1
    }
}

/// Computes the bitmask for the final word of the vector
fn mask_for_bits(bits: uint) -> u32 {
    // Note especially that a perfect multiple of u32::BITS should mask all 1s.
    !0u32 >> (u32::BITS - bits % u32::BITS) % u32::BITS
}

impl Bitv {
    /// Applies the given operation to the blocks of self and other, and sets
    /// self to be the result. This relies on the caller not to corrupt the
    /// last word.
    #[inline]
    fn process<F>(&mut self, other: &Bitv, mut op: F) -> bool where F: FnMut(u32, u32) -> u32 {
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
    /// `Bitv`. As per (3), it's assumed to be all 0s. This method fixes it up.
    fn fix_last_block(&mut self) {
        let extra_bits = self.len() % u32::BITS;
        if extra_bits > 0 {
            let mask = (1 << extra_bits) - 1;
            let storage_len = self.storage.len();
            self.storage[storage_len - 1] &= mask;
        }
    }

    /// Creates an empty `Bitv`.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::Bitv;
    /// let mut bv = Bitv::new();
    /// ```
    #[stable]
    pub fn new() -> Bitv {
        Bitv { storage: Vec::new(), nbits: 0 }
    }

    /// Creates a `Bitv` that holds `nbits` elements, setting each element
    /// to `bit`.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::Bitv;
    ///
    /// let mut bv = Bitv::from_elem(10u, false);
    /// assert_eq!(bv.len(), 10u);
    /// for x in bv.iter() {
    ///     assert_eq!(x, false);
    /// }
    /// ```
    pub fn from_elem(nbits: uint, bit: bool) -> Bitv {
        let nblocks = blocks_for_bits(nbits);
        let mut bitv = Bitv {
            storage: repeat(if bit { !0u32 } else { 0u32 }).take(nblocks).collect(),
            nbits: nbits
        };
        bitv.fix_last_block();
        bitv
    }

    /// Constructs a new, empty `Bitv` with the specified capacity.
    ///
    /// The bitvector will be able to hold at least `capacity` bits without
    /// reallocating. If `capacity` is 0, it will not allocate.
    ///
    /// It is important to note that this function does not specify the
    /// *length* of the returned bitvector, but only the *capacity*.
    #[stable]
    pub fn with_capacity(nbits: uint) -> Bitv {
        Bitv {
            storage: Vec::with_capacity(blocks_for_bits(nbits)),
            nbits: 0,
        }
    }

    /// Transforms a byte-vector into a `Bitv`. Each byte becomes eight bits,
    /// with the most significant bits of each byte coming first. Each
    /// bit becomes `true` if equal to 1 or `false` if equal to 0.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::Bitv;
    ///
    /// let bv = Bitv::from_bytes(&[0b10100000, 0b00010010]);
    /// assert!(bv.eq_vec(&[true, false, true, false,
    ///                     false, false, false, false,
    ///                     false, false, false, true,
    ///                     false, false, true, false]));
    /// ```
    pub fn from_bytes(bytes: &[u8]) -> Bitv {
        let len = bytes.len().checked_mul(u8::BITS).expect("capacity overflow");
        let mut bitv = Bitv::with_capacity(len);
        let complete_words = bytes.len() / 4;
        let extra_bytes = bytes.len() % 4;

        bitv.nbits = len;

        for i in range(0, complete_words) {
            bitv.storage.push(
                ((reverse_bits(bytes[i * 4 + 0]) as u32) << 0) |
                ((reverse_bits(bytes[i * 4 + 1]) as u32) << 8) |
                ((reverse_bits(bytes[i * 4 + 2]) as u32) << 16) |
                ((reverse_bits(bytes[i * 4 + 3]) as u32) << 24)
            );
        }

        if extra_bytes > 0 {
            let mut last_word = 0u32;
            for (i, &byte) in bytes[(complete_words*4)..].iter().enumerate() {
                last_word |= (reverse_bits(byte) as u32) << (i * 8);
            }
            bitv.storage.push(last_word);
        }

        bitv
    }

    /// Creates a `Bitv` of the specified length where the value at each index
    /// is `f(index)`.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::Bitv;
    ///
    /// let bv = Bitv::from_fn(5, |i| { i % 2 == 0 });
    /// assert!(bv.eq_vec(&[true, false, true, false, true]));
    /// ```
    pub fn from_fn<F>(len: uint, mut f: F) -> Bitv where F: FnMut(uint) -> bool {
        let mut bitv = Bitv::from_elem(len, false);
        for i in range(0u, len) {
            bitv.set(i, f(i));
        }
        bitv
    }

    /// Retrieves the value at index `i`, or `None` if the index is out of bounds.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::Bitv;
    ///
    /// let bv = Bitv::from_bytes(&[0b01100000]);
    /// assert_eq!(bv.get(0), Some(false));
    /// assert_eq!(bv.get(1), Some(true));
    /// assert_eq!(bv.get(100), None);
    ///
    /// // Can also use array indexing
    /// assert_eq!(bv[1], true);
    /// ```
    #[inline]
    #[stable]
    pub fn get(&self, i: uint) -> Option<bool> {
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
    /// use std::collections::Bitv;
    ///
    /// let mut bv = Bitv::from_elem(5, false);
    /// bv.set(3, true);
    /// assert_eq!(bv[3], true);
    /// ```
    #[inline]
    #[unstable = "panic semantics are likely to change in the future"]
    pub fn set(&mut self, i: uint, x: bool) {
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
    /// use std::collections::Bitv;
    ///
    /// let before = 0b01100000;
    /// let after  = 0b11111111;
    ///
    /// let mut bv = Bitv::from_bytes(&[before]);
    /// bv.set_all();
    /// assert_eq!(bv, Bitv::from_bytes(&[after]));
    /// ```
    #[inline]
    pub fn set_all(&mut self) {
        for w in self.storage.iter_mut() { *w = !0u32; }
        self.fix_last_block();
    }

    /// Flips all bits.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::Bitv;
    ///
    /// let before = 0b01100000;
    /// let after  = 0b10011111;
    ///
    /// let mut bv = Bitv::from_bytes(&[before]);
    /// bv.negate();
    /// assert_eq!(bv, Bitv::from_bytes(&[after]));
    /// ```
    #[inline]
    pub fn negate(&mut self) {
        for w in self.storage.iter_mut() { *w = !*w; }
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
    /// use std::collections::Bitv;
    ///
    /// let a   = 0b01100100;
    /// let b   = 0b01011010;
    /// let res = 0b01111110;
    ///
    /// let mut a = Bitv::from_bytes(&[a]);
    /// let b = Bitv::from_bytes(&[b]);
    ///
    /// assert!(a.union(&b));
    /// assert_eq!(a, Bitv::from_bytes(&[res]));
    /// ```
    #[inline]
    pub fn union(&mut self, other: &Bitv) -> bool {
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
    /// use std::collections::Bitv;
    ///
    /// let a   = 0b01100100;
    /// let b   = 0b01011010;
    /// let res = 0b01000000;
    ///
    /// let mut a = Bitv::from_bytes(&[a]);
    /// let b = Bitv::from_bytes(&[b]);
    ///
    /// assert!(a.intersect(&b));
    /// assert_eq!(a, Bitv::from_bytes(&[res]));
    /// ```
    #[inline]
    pub fn intersect(&mut self, other: &Bitv) -> bool {
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
    /// use std::collections::Bitv;
    ///
    /// let a   = 0b01100100;
    /// let b   = 0b01011010;
    /// let a_b = 0b00100100; // a - b
    /// let b_a = 0b00011010; // b - a
    ///
    /// let mut bva = Bitv::from_bytes(&[a]);
    /// let bvb = Bitv::from_bytes(&[b]);
    ///
    /// assert!(bva.difference(&bvb));
    /// assert_eq!(bva, Bitv::from_bytes(&[a_b]));
    ///
    /// let bva = Bitv::from_bytes(&[a]);
    /// let mut bvb = Bitv::from_bytes(&[b]);
    ///
    /// assert!(bvb.difference(&bva));
    /// assert_eq!(bvb, Bitv::from_bytes(&[b_a]));
    /// ```
    #[inline]
    pub fn difference(&mut self, other: &Bitv) -> bool {
        self.process(other, |w1, w2| w1 & !w2)
    }

    /// Returns `true` if all bits are 1.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::Bitv;
    ///
    /// let mut bv = Bitv::from_elem(5, true);
    /// assert_eq!(bv.all(), true);
    ///
    /// bv.set(1, false);
    /// assert_eq!(bv.all(), false);
    /// ```
    pub fn all(&self) -> bool {
        let mut last_word = !0u32;
        // Check that every block but the last is all-ones...
        self.blocks().all(|elem| {
            let tmp = last_word;
            last_word = elem;
            tmp == !0u32
        // and then check the last one has enough ones
        }) && (last_word == mask_for_bits(self.nbits))
    }

    /// Returns an iterator over the elements of the vector in order.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::Bitv;
    ///
    /// let bv = Bitv::from_bytes(&[0b01110100, 0b10010010]);
    /// assert_eq!(bv.iter().filter(|x| *x).count(), 7);
    /// ```
    #[inline]
    #[stable]
    pub fn iter(&self) -> Iter {
        Iter { bitv: self, next_idx: 0, end_idx: self.nbits }
    }

    /// Returns `true` if all bits are 0.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::Bitv;
    ///
    /// let mut bv = Bitv::from_elem(10, false);
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
    /// use std::collections::Bitv;
    ///
    /// let mut bv = Bitv::from_elem(10, false);
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
    /// `Bitv` becomes the high-order bit of the first byte. If the
    /// size of the `Bitv` is not a multiple of eight then trailing bits
    /// will be filled-in with `false`.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::Bitv;
    ///
    /// let mut bv = Bitv::from_elem(3, true);
    /// bv.set(1, false);
    ///
    /// assert_eq!(bv.to_bytes(), vec!(0b10100000));
    ///
    /// let mut bv = Bitv::from_elem(9, false);
    /// bv.set(2, true);
    /// bv.set(8, true);
    ///
    /// assert_eq!(bv.to_bytes(), vec!(0b00100000, 0b10000000));
    /// ```
    pub fn to_bytes(&self) -> Vec<u8> {
        fn bit(bitv: &Bitv, byte: uint, bit: uint) -> u8 {
            let offset = byte * 8 + bit;
            if offset >= bitv.nbits {
                0
            } else {
                (bitv[offset] as u8) << (7 - bit)
            }
        }

        let len = self.nbits/8 +
                  if self.nbits % 8 == 0 { 0 } else { 1 };
        range(0, len).map(|i|
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

    /// Compares a `Bitv` to a slice of `bool`s.
    /// Both the `Bitv` and slice must have the same length.
    ///
    /// # Panics
    ///
    /// Panics if the `Bitv` and slice are of different length.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::Bitv;
    ///
    /// let bv = Bitv::from_bytes(&[0b10100000]);
    ///
    /// assert!(bv.eq_vec(&[true, false, true, false,
    ///                     false, false, false, false]));
    /// ```
    pub fn eq_vec(&self, v: &[bool]) -> bool {
        assert_eq!(self.nbits, v.len());
        iter::order::eq(self.iter(), v.iter().cloned())
    }

    /// Shortens a `Bitv`, dropping excess elements.
    ///
    /// If `len` is greater than the vector's current length, this has no
    /// effect.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::Bitv;
    ///
    /// let mut bv = Bitv::from_bytes(&[0b01001011]);
    /// bv.truncate(2);
    /// assert!(bv.eq_vec(&[false, true]));
    /// ```
    #[stable]
    pub fn truncate(&mut self, len: uint) {
        if len < self.len() {
            self.nbits = len;
            // This fixes (2).
            self.storage.truncate(blocks_for_bits(len));
            self.fix_last_block();
        }
    }

    /// Reserves capacity for at least `additional` more bits to be inserted in the given
    /// `Bitv`. The collection may reserve more space to avoid frequent reallocations.
    ///
    /// # Panics
    ///
    /// Panics if the new capacity overflows `uint`.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::Bitv;
    ///
    /// let mut bv = Bitv::from_elem(3, false);
    /// bv.reserve(10);
    /// assert_eq!(bv.len(), 3);
    /// assert!(bv.capacity() >= 13);
    /// ```
    #[stable]
    pub fn reserve(&mut self, additional: uint) {
        let desired_cap = self.len().checked_add(additional).expect("capacity overflow");
        let storage_len = self.storage.len();
        if desired_cap > self.capacity() {
            self.storage.reserve(blocks_for_bits(desired_cap) - storage_len);
        }
    }

    /// Reserves the minimum capacity for exactly `additional` more bits to be inserted in the
    /// given `Bitv`. Does nothing if the capacity is already sufficient.
    ///
    /// Note that the allocator may give the collection more space than it requests. Therefore
    /// capacity can not be relied upon to be precisely minimal. Prefer `reserve` if future
    /// insertions are expected.
    ///
    /// # Panics
    ///
    /// Panics if the new capacity overflows `uint`.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::Bitv;
    ///
    /// let mut bv = Bitv::from_elem(3, false);
    /// bv.reserve(10);
    /// assert_eq!(bv.len(), 3);
    /// assert!(bv.capacity() >= 13);
    /// ```
    #[stable]
    pub fn reserve_exact(&mut self, additional: uint) {
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
    /// use std::collections::Bitv;
    ///
    /// let mut bv = Bitv::new();
    /// bv.reserve(10);
    /// assert!(bv.capacity() >= 10);
    /// ```
    #[inline]
    #[stable]
    pub fn capacity(&self) -> uint {
        self.storage.capacity().checked_mul(u32::BITS).unwrap_or(uint::MAX)
    }

    /// Grows the `Bitv` in-place, adding `n` copies of `value` to the `Bitv`.
    ///
    /// # Panics
    ///
    /// Panics if the new len overflows a `uint`.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::Bitv;
    ///
    /// let mut bv = Bitv::from_bytes(&[0b01001011]);
    /// bv.grow(2, true);
    /// assert_eq!(bv.len(), 10);
    /// assert_eq!(bv.to_bytes(), vec!(0b01001011, 0b11000000));
    /// ```
    pub fn grow(&mut self, n: uint, value: bool) {
        // Note: we just bulk set all the bits in the last word in this fn in multiple places
        // which is technically wrong if not all of these bits are to be used. However, at the end
        // of this fn we call `fix_last_block` at the end of this fn, which should fix this.

        let new_nbits = self.nbits.checked_add(n).expect("capacity overflow");
        let new_nblocks = blocks_for_bits(new_nbits);
        let full_value = if value { !0 } else { 0 };

        // Correct the old tail word, setting or clearing formerly unused bits
        let old_last_word = blocks_for_bits(self.nbits) - 1;
        if self.nbits % u32::BITS > 0 {
            let mask = mask_for_bits(self.nbits);
            if value {
                self.storage[old_last_word] |= !mask;
            } else {
                // Extra bits are already zero by invariant.
            }
        }

        // Fill in words after the old tail word
        let stop_idx = cmp::min(self.storage.len(), new_nblocks);
        for idx in range(old_last_word + 1, stop_idx) {
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

    /// Removes the last bit from the Bitv, and returns it. Returns None if the Bitv is empty.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::Bitv;
    ///
    /// let mut bv = Bitv::from_bytes(&[0b01001001]);
    /// assert_eq!(bv.pop(), Some(true));
    /// assert_eq!(bv.pop(), Some(false));
    /// assert_eq!(bv.len(), 6);
    /// ```
    #[stable]
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
    /// use std::collections::Bitv;
    ///
    /// let mut bv = Bitv::new();
    /// bv.push(true);
    /// bv.push(false);
    /// assert!(bv.eq_vec(&[true, false]));
    /// ```
    #[stable]
    pub fn push(&mut self, elem: bool) {
        if self.nbits % u32::BITS == 0 {
            self.storage.push(0);
        }
        let insert_pos = self.nbits;
        self.nbits = self.nbits.checked_add(1).expect("Capacity overflow");
        self.set(insert_pos, elem);
    }

    /// Return the total number of bits in this vector
    #[inline]
    #[stable]
    pub fn len(&self) -> uint { self.nbits }

    /// Returns true if there are no bits in this vector
    #[inline]
    #[stable]
    pub fn is_empty(&self) -> bool { self.len() == 0 }

    /// Clears all bits in this vector.
    #[inline]
    #[stable]
    pub fn clear(&mut self) {
        for w in self.storage.iter_mut() { *w = 0u32; }
    }
}

#[stable]
impl Default for Bitv {
    #[inline]
    fn default() -> Bitv { Bitv::new() }
}

#[stable]
impl FromIterator<bool> for Bitv {
    fn from_iter<I:Iterator<Item=bool>>(iterator: I) -> Bitv {
        let mut ret = Bitv::new();
        ret.extend(iterator);
        ret
    }
}

#[stable]
impl Extend<bool> for Bitv {
    #[inline]
    fn extend<I: Iterator<Item=bool>>(&mut self, mut iterator: I) {
        let (min, _) = iterator.size_hint();
        self.reserve(min);
        for element in iterator {
            self.push(element)
        }
    }
}

#[stable]
impl Clone for Bitv {
    #[inline]
    fn clone(&self) -> Bitv {
        Bitv { storage: self.storage.clone(), nbits: self.nbits }
    }

    #[inline]
    fn clone_from(&mut self, source: &Bitv) {
        self.nbits = source.nbits;
        self.storage.clone_from(&source.storage);
    }
}

#[stable]
impl PartialOrd for Bitv {
    #[inline]
    fn partial_cmp(&self, other: &Bitv) -> Option<Ordering> {
        iter::order::partial_cmp(self.iter(), other.iter())
    }
}

#[stable]
impl Ord for Bitv {
    #[inline]
    fn cmp(&self, other: &Bitv) -> Ordering {
        iter::order::cmp(self.iter(), other.iter())
    }
}

#[stable]
impl fmt::Show for Bitv {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        for bit in self.iter() {
            try!(write!(fmt, "{}", if bit { 1u32 } else { 0u32 }));
        }
        Ok(())
    }
}

#[stable]
impl<S: hash::Writer + hash::Hasher> hash::Hash<S> for Bitv {
    fn hash(&self, state: &mut S) {
        self.nbits.hash(state);
        for elem in self.blocks() {
            elem.hash(state);
        }
    }
}

#[stable]
impl cmp::PartialEq for Bitv {
    #[inline]
    fn eq(&self, other: &Bitv) -> bool {
        if self.nbits != other.nbits {
            return false;
        }
        self.blocks().zip(other.blocks()).all(|(w1, w2)| w1 == w2)
    }
}

#[stable]
impl cmp::Eq for Bitv {}

/// An iterator for `Bitv`.
#[stable]
#[derive(Clone)]
pub struct Iter<'a> {
    bitv: &'a Bitv,
    next_idx: uint,
    end_idx: uint,
}

#[stable]
impl<'a> Iterator for Iter<'a> {
    type Item = bool;

    #[inline]
    fn next(&mut self) -> Option<bool> {
        if self.next_idx != self.end_idx {
            let idx = self.next_idx;
            self.next_idx += 1;
            Some(self.bitv[idx])
        } else {
            None
        }
    }

    fn size_hint(&self) -> (uint, Option<uint>) {
        let rem = self.end_idx - self.next_idx;
        (rem, Some(rem))
    }
}

#[stable]
impl<'a> DoubleEndedIterator for Iter<'a> {
    #[inline]
    fn next_back(&mut self) -> Option<bool> {
        if self.next_idx != self.end_idx {
            self.end_idx -= 1;
            Some(self.bitv[self.end_idx])
        } else {
            None
        }
    }
}

#[stable]
impl<'a> ExactSizeIterator for Iter<'a> {}

#[stable]
impl<'a> RandomAccessIterator for Iter<'a> {
    #[inline]
    fn indexable(&self) -> uint {
        self.end_idx - self.next_idx
    }

    #[inline]
    fn idx(&mut self, index: uint) -> Option<bool> {
        if index >= self.indexable() {
            None
        } else {
            Some(self.bitv[index])
        }
    }
}

/// An implementation of a set using a bit vector as an underlying
/// representation for holding unsigned numerical elements.
///
/// It should also be noted that the amount of storage necessary for holding a
/// set of objects is proportional to the maximum of the objects when viewed
/// as a `uint`.
///
/// # Examples
///
/// ```
/// use std::collections::{BitvSet, Bitv};
///
/// // It's a regular set
/// let mut s = BitvSet::new();
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
/// // Can initialize from a `Bitv`
/// let other = BitvSet::from_bitv(Bitv::from_bytes(&[0b11010000]));
///
/// s.union_with(&other);
///
/// // Print 0, 1, 3 in some order
/// for x in s.iter() {
///     println!("{}", x);
/// }
///
/// // Can convert back to a `Bitv`
/// let bv: Bitv = s.into_bitv();
/// assert!(bv[3]);
/// ```
#[derive(Clone)]
#[stable]
pub struct BitvSet {
    bitv: Bitv,
}

#[stable]
impl Default for BitvSet {
    #[inline]
    fn default() -> BitvSet { BitvSet::new() }
}

#[stable]
impl FromIterator<uint> for BitvSet {
    fn from_iter<I:Iterator<Item=uint>>(iterator: I) -> BitvSet {
        let mut ret = BitvSet::new();
        ret.extend(iterator);
        ret
    }
}

#[stable]
impl Extend<uint> for BitvSet {
    #[inline]
    fn extend<I: Iterator<Item=uint>>(&mut self, mut iterator: I) {
        for i in iterator {
            self.insert(i);
        }
    }
}

#[stable]
impl PartialOrd for BitvSet {
    #[inline]
    fn partial_cmp(&self, other: &BitvSet) -> Option<Ordering> {
        let (a_iter, b_iter) = match_words(self.get_ref(), other.get_ref());
        iter::order::partial_cmp(a_iter, b_iter)
    }
}

#[stable]
impl Ord for BitvSet {
    #[inline]
    fn cmp(&self, other: &BitvSet) -> Ordering {
        let (a_iter, b_iter) = match_words(self.get_ref(), other.get_ref());
        iter::order::cmp(a_iter, b_iter)
    }
}

#[stable]
impl cmp::PartialEq for BitvSet {
    #[inline]
    fn eq(&self, other: &BitvSet) -> bool {
        let (a_iter, b_iter) = match_words(self.get_ref(), other.get_ref());
        iter::order::eq(a_iter, b_iter)
    }
}

#[stable]
impl cmp::Eq for BitvSet {}

impl BitvSet {
    /// Creates a new empty `BitvSet`.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::BitvSet;
    ///
    /// let mut s = BitvSet::new();
    /// ```
    #[inline]
    #[stable]
    pub fn new() -> BitvSet {
        BitvSet { bitv: Bitv::new() }
    }

    /// Creates a new `BitvSet` with initially no contents, able to
    /// hold `nbits` elements without resizing.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::BitvSet;
    ///
    /// let mut s = BitvSet::with_capacity(100);
    /// assert!(s.capacity() >= 100);
    /// ```
    #[inline]
    #[stable]
    pub fn with_capacity(nbits: uint) -> BitvSet {
        let bitv = Bitv::from_elem(nbits, false);
        BitvSet::from_bitv(bitv)
    }

    /// Creates a new `BitvSet` from the given bit vector.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::{Bitv, BitvSet};
    ///
    /// let bv = Bitv::from_bytes(&[0b01100000]);
    /// let s = BitvSet::from_bitv(bv);
    ///
    /// // Print 1, 2 in arbitrary order
    /// for x in s.iter() {
    ///     println!("{}", x);
    /// }
    /// ```
    #[inline]
    pub fn from_bitv(bitv: Bitv) -> BitvSet {
        BitvSet { bitv: bitv }
    }

    /// Returns the capacity in bits for this bit vector. Inserting any
    /// element less than this amount will not trigger a resizing.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::BitvSet;
    ///
    /// let mut s = BitvSet::with_capacity(100);
    /// assert!(s.capacity() >= 100);
    /// ```
    #[inline]
    #[stable]
    pub fn capacity(&self) -> uint {
        self.bitv.capacity()
    }

    /// Reserves capacity for the given `BitvSet` to contain `len` distinct elements. In the case
    /// of `BitvSet` this means reallocations will not occur as long as all inserted elements
    /// are less than `len`.
    ///
    /// The collection may reserve more space to avoid frequent reallocations.
    ///
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::BitvSet;
    ///
    /// let mut s = BitvSet::new();
    /// s.reserve_len(10);
    /// assert!(s.capacity() >= 10);
    /// ```
    #[stable]
    pub fn reserve_len(&mut self, len: uint) {
        let cur_len = self.bitv.len();
        if len >= cur_len {
            self.bitv.reserve(len - cur_len);
        }
    }

    /// Reserves the minimum capacity for the given `BitvSet` to contain `len` distinct elements.
    /// In the case of `BitvSet` this means reallocations will not occur as long as all inserted
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
    /// use std::collections::BitvSet;
    ///
    /// let mut s = BitvSet::new();
    /// s.reserve_len_exact(10);
    /// assert!(s.capacity() >= 10);
    /// ```
    #[stable]
    pub fn reserve_len_exact(&mut self, len: uint) {
        let cur_len = self.bitv.len();
        if len >= cur_len {
            self.bitv.reserve_exact(len - cur_len);
        }
    }


    /// Consumes this set to return the underlying bit vector.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::BitvSet;
    ///
    /// let mut s = BitvSet::new();
    /// s.insert(0);
    /// s.insert(3);
    ///
    /// let bv = s.into_bitv();
    /// assert!(bv[0]);
    /// assert!(bv[3]);
    /// ```
    #[inline]
    pub fn into_bitv(self) -> Bitv {
        self.bitv
    }

    /// Returns a reference to the underlying bit vector.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::BitvSet;
    ///
    /// let mut s = BitvSet::new();
    /// s.insert(0);
    ///
    /// let bv = s.get_ref();
    /// assert_eq!(bv[0], true);
    /// ```
    #[inline]
    pub fn get_ref(&self) -> &Bitv {
        &self.bitv
    }

    #[inline]
    fn other_op<F>(&mut self, other: &BitvSet, mut f: F) where F: FnMut(u32, u32) -> u32 {
        // Unwrap Bitvs
        let self_bitv = &mut self.bitv;
        let other_bitv = &other.bitv;

        let self_len = self_bitv.len();
        let other_len = other_bitv.len();

        // Expand the vector if necessary
        if self_len < other_len {
            self_bitv.grow(other_len - self_len, false);
        }

        // virtually pad other with 0's for equal lengths
        let mut other_words = {
            let (_, result) = match_words(self_bitv, other_bitv);
            result
        };

        // Apply values found in other
        for (i, w) in other_words {
            let old = self_bitv.storage[i];
            let new = f(old, w);
            self_bitv.storage[i] = new;
        }
    }

    /// Truncates the underlying vector to the least length required.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::BitvSet;
    ///
    /// let mut s = BitvSet::new();
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
    #[stable]
    pub fn shrink_to_fit(&mut self) {
        let bitv = &mut self.bitv;
        // Obtain original length
        let old_len = bitv.storage.len();
        // Obtain coarse trailing zero length
        let n = bitv.storage.iter().rev().take_while(|&&n| n == 0).count();
        // Truncate
        let trunc_len = cmp::max(old_len - n, 1);
        bitv.storage.truncate(trunc_len);
        bitv.nbits = trunc_len * u32::BITS;
    }

    /// Iterator over each u32 stored in the `BitvSet`.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::{Bitv, BitvSet};
    ///
    /// let s = BitvSet::from_bitv(Bitv::from_bytes(&[0b01001010]));
    ///
    /// // Print 1, 4, 6 in arbitrary order
    /// for x in s.iter() {
    ///     println!("{}", x);
    /// }
    /// ```
    #[inline]
    #[stable]
    pub fn iter(&self) -> bitv_set::Iter {
        SetIter {set: self, next_idx: 0u}
    }

    /// Iterator over each u32 stored in `self` union `other`.
    /// See [union_with](#method.union_with) for an efficient in-place version.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::{Bitv, BitvSet};
    ///
    /// let a = BitvSet::from_bitv(Bitv::from_bytes(&[0b01101000]));
    /// let b = BitvSet::from_bitv(Bitv::from_bytes(&[0b10100000]));
    ///
    /// // Print 0, 1, 2, 4 in arbitrary order
    /// for x in a.union(&b) {
    ///     println!("{}", x);
    /// }
    /// ```
    #[inline]
    #[stable]
    pub fn union<'a>(&'a self, other: &'a BitvSet) -> Union<'a> {
        fn or(w1: u32, w2: u32) -> u32 { w1 | w2 }

        Union(TwoBitPositions {
            set: self,
            other: other,
            merge: or,
            current_word: 0u32,
            next_idx: 0u
        })
    }

    /// Iterator over each uint stored in `self` intersect `other`.
    /// See [intersect_with](#method.intersect_with) for an efficient in-place version.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::{Bitv, BitvSet};
    ///
    /// let a = BitvSet::from_bitv(Bitv::from_bytes(&[0b01101000]));
    /// let b = BitvSet::from_bitv(Bitv::from_bytes(&[0b10100000]));
    ///
    /// // Print 2
    /// for x in a.intersection(&b) {
    ///     println!("{}", x);
    /// }
    /// ```
    #[inline]
    #[stable]
    pub fn intersection<'a>(&'a self, other: &'a BitvSet) -> Intersection<'a> {
        fn bitand(w1: u32, w2: u32) -> u32 { w1 & w2 }
        let min = cmp::min(self.bitv.len(), other.bitv.len());
        Intersection(TwoBitPositions {
            set: self,
            other: other,
            merge: bitand,
            current_word: 0u32,
            next_idx: 0
        }.take(min))
    }

    /// Iterator over each uint stored in the `self` setminus `other`.
    /// See [difference_with](#method.difference_with) for an efficient in-place version.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::{BitvSet, Bitv};
    ///
    /// let a = BitvSet::from_bitv(Bitv::from_bytes(&[0b01101000]));
    /// let b = BitvSet::from_bitv(Bitv::from_bytes(&[0b10100000]));
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
    #[stable]
    pub fn difference<'a>(&'a self, other: &'a BitvSet) -> Difference<'a> {
        fn diff(w1: u32, w2: u32) -> u32 { w1 & !w2 }

        Difference(TwoBitPositions {
            set: self,
            other: other,
            merge: diff,
            current_word: 0u32,
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
    /// use std::collections::{BitvSet, Bitv};
    ///
    /// let a = BitvSet::from_bitv(Bitv::from_bytes(&[0b01101000]));
    /// let b = BitvSet::from_bitv(Bitv::from_bytes(&[0b10100000]));
    ///
    /// // Print 0, 1, 4 in arbitrary order
    /// for x in a.symmetric_difference(&b) {
    ///     println!("{}", x);
    /// }
    /// ```
    #[inline]
    #[stable]
    pub fn symmetric_difference<'a>(&'a self, other: &'a BitvSet) -> SymmetricDifference<'a> {
        fn bitxor(w1: u32, w2: u32) -> u32 { w1 ^ w2 }

        SymmetricDifference(TwoBitPositions {
            set: self,
            other: other,
            merge: bitxor,
            current_word: 0u32,
            next_idx: 0
        })
    }

    /// Unions in-place with the specified other bit vector.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::{BitvSet, Bitv};
    ///
    /// let a   = 0b01101000;
    /// let b   = 0b10100000;
    /// let res = 0b11101000;
    ///
    /// let mut a = BitvSet::from_bitv(Bitv::from_bytes(&[a]));
    /// let b = BitvSet::from_bitv(Bitv::from_bytes(&[b]));
    /// let res = BitvSet::from_bitv(Bitv::from_bytes(&[res]));
    ///
    /// a.union_with(&b);
    /// assert_eq!(a, res);
    /// ```
    #[inline]
    pub fn union_with(&mut self, other: &BitvSet) {
        self.other_op(other, |w1, w2| w1 | w2);
    }

    /// Intersects in-place with the specified other bit vector.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::{BitvSet, Bitv};
    ///
    /// let a   = 0b01101000;
    /// let b   = 0b10100000;
    /// let res = 0b00100000;
    ///
    /// let mut a = BitvSet::from_bitv(Bitv::from_bytes(&[a]));
    /// let b = BitvSet::from_bitv(Bitv::from_bytes(&[b]));
    /// let res = BitvSet::from_bitv(Bitv::from_bytes(&[res]));
    ///
    /// a.intersect_with(&b);
    /// assert_eq!(a, res);
    /// ```
    #[inline]
    pub fn intersect_with(&mut self, other: &BitvSet) {
        self.other_op(other, |w1, w2| w1 & w2);
    }

    /// Makes this bit vector the difference with the specified other bit vector
    /// in-place.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::{BitvSet, Bitv};
    ///
    /// let a   = 0b01101000;
    /// let b   = 0b10100000;
    /// let a_b = 0b01001000; // a - b
    /// let b_a = 0b10000000; // b - a
    ///
    /// let mut bva = BitvSet::from_bitv(Bitv::from_bytes(&[a]));
    /// let bvb = BitvSet::from_bitv(Bitv::from_bytes(&[b]));
    /// let bva_b = BitvSet::from_bitv(Bitv::from_bytes(&[a_b]));
    /// let bvb_a = BitvSet::from_bitv(Bitv::from_bytes(&[b_a]));
    ///
    /// bva.difference_with(&bvb);
    /// assert_eq!(bva, bva_b);
    ///
    /// let bva = BitvSet::from_bitv(Bitv::from_bytes(&[a]));
    /// let mut bvb = BitvSet::from_bitv(Bitv::from_bytes(&[b]));
    ///
    /// bvb.difference_with(&bva);
    /// assert_eq!(bvb, bvb_a);
    /// ```
    #[inline]
    pub fn difference_with(&mut self, other: &BitvSet) {
        self.other_op(other, |w1, w2| w1 & !w2);
    }

    /// Makes this bit vector the symmetric difference with the specified other
    /// bit vector in-place.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::{BitvSet, Bitv};
    ///
    /// let a   = 0b01101000;
    /// let b   = 0b10100000;
    /// let res = 0b11001000;
    ///
    /// let mut a = BitvSet::from_bitv(Bitv::from_bytes(&[a]));
    /// let b = BitvSet::from_bitv(Bitv::from_bytes(&[b]));
    /// let res = BitvSet::from_bitv(Bitv::from_bytes(&[res]));
    ///
    /// a.symmetric_difference_with(&b);
    /// assert_eq!(a, res);
    /// ```
    #[inline]
    pub fn symmetric_difference_with(&mut self, other: &BitvSet) {
        self.other_op(other, |w1, w2| w1 ^ w2);
    }

    /// Return the number of set bits in this set.
    #[inline]
    #[stable]
    pub fn len(&self) -> uint  {
        self.bitv.blocks().fold(0, |acc, n| acc + n.count_ones())
    }

    /// Returns whether there are no bits set in this set
    #[inline]
    #[stable]
    pub fn is_empty(&self) -> bool {
        self.bitv.none()
    }

    /// Clears all bits in this set
    #[inline]
    #[stable]
    pub fn clear(&mut self) {
        self.bitv.clear();
    }

    /// Returns `true` if this set contains the specified integer.
    #[inline]
    #[stable]
    pub fn contains(&self, value: &uint) -> bool {
        let bitv = &self.bitv;
        *value < bitv.nbits && bitv[*value]
    }

    /// Returns `true` if the set has no elements in common with `other`.
    /// This is equivalent to checking for an empty intersection.
    #[inline]
    #[stable]
    pub fn is_disjoint(&self, other: &BitvSet) -> bool {
        self.intersection(other).next().is_none()
    }

    /// Returns `true` if the set is a subset of another.
    #[inline]
    #[stable]
    pub fn is_subset(&self, other: &BitvSet) -> bool {
        let self_bitv = &self.bitv;
        let other_bitv = &other.bitv;
        let other_blocks = blocks_for_bits(other_bitv.len());

        // Check that `self` intersect `other` is self
        self_bitv.blocks().zip(other_bitv.blocks()).all(|(w1, w2)| w1 & w2 == w1) &&
        // Make sure if `self` has any more blocks than `other`, they're all 0
        self_bitv.blocks().skip(other_blocks).all(|w| w == 0)
    }

    /// Returns `true` if the set is a superset of another.
    #[inline]
    #[stable]
    pub fn is_superset(&self, other: &BitvSet) -> bool {
        other.is_subset(self)
    }

    /// Adds a value to the set. Returns `true` if the value was not already
    /// present in the set.
    #[stable]
    pub fn insert(&mut self, value: uint) -> bool {
        if self.contains(&value) {
            return false;
        }

        // Ensure we have enough space to hold the new element
        let len = self.bitv.len();
        if value >= len {
            self.bitv.grow(value - len + 1, false)
        }

        self.bitv.set(value, true);
        return true;
    }

    /// Removes a value from the set. Returns `true` if the value was
    /// present in the set.
    #[stable]
    pub fn remove(&mut self, value: &uint) -> bool {
        if !self.contains(value) {
            return false;
        }

        self.bitv.set(*value, false);

        return true;
    }
}

impl fmt::Show for BitvSet {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        try!(write!(fmt, "BitvSet {{"));
        let mut first = true;
        for n in self.iter() {
            if !first {
                try!(write!(fmt, ", "));
            }
            try!(write!(fmt, "{:?}", n));
            first = false;
        }
        write!(fmt, "}}")
    }
}

impl<S: hash::Writer + hash::Hasher> hash::Hash<S> for BitvSet {
    fn hash(&self, state: &mut S) {
        for pos in self.iter() {
            pos.hash(state);
        }
    }
}

/// An iterator for `BitvSet`.
#[derive(Clone)]
#[stable]
pub struct SetIter<'a> {
    set: &'a BitvSet,
    next_idx: uint
}

/// An iterator combining two `BitvSet` iterators.
#[derive(Clone)]
struct TwoBitPositions<'a> {
    set: &'a BitvSet,
    other: &'a BitvSet,
    merge: fn(u32, u32) -> u32,
    current_word: u32,
    next_idx: uint
}

#[stable]
pub struct Union<'a>(TwoBitPositions<'a>);
#[stable]
pub struct Intersection<'a>(Take<TwoBitPositions<'a>>);
#[stable]
pub struct Difference<'a>(TwoBitPositions<'a>);
#[stable]
pub struct SymmetricDifference<'a>(TwoBitPositions<'a>);

#[stable]
impl<'a> Iterator for SetIter<'a> {
    type Item = uint;

    fn next(&mut self) -> Option<uint> {
        while self.next_idx < self.set.bitv.len() {
            let idx = self.next_idx;
            self.next_idx += 1;

            if self.set.contains(&idx) {
                return Some(idx);
            }
        }

        return None;
    }

    #[inline]
    fn size_hint(&self) -> (uint, Option<uint>) {
        (0, Some(self.set.bitv.len() - self.next_idx))
    }
}

#[stable]
impl<'a> Iterator for TwoBitPositions<'a> {
    type Item = uint;

    fn next(&mut self) -> Option<uint> {
        while self.next_idx < self.set.bitv.len() ||
              self.next_idx < self.other.bitv.len() {
            let bit_idx = self.next_idx % u32::BITS;
            if bit_idx == 0 {
                let s_bitv = &self.set.bitv;
                let o_bitv = &self.other.bitv;
                // Merging the two words is a bit of an awkward dance since
                // one Bitv might be longer than the other
                let word_idx = self.next_idx / u32::BITS;
                let w1 = if word_idx < s_bitv.storage.len() {
                             s_bitv.storage[word_idx]
                         } else { 0 };
                let w2 = if word_idx < o_bitv.storage.len() {
                             o_bitv.storage[word_idx]
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
    fn size_hint(&self) -> (uint, Option<uint>) {
        let cap = cmp::max(self.set.bitv.len(), self.other.bitv.len());
        (0, Some(cap - self.next_idx))
    }
}

#[stable]
impl<'a> Iterator for Union<'a> {
    type Item = uint;

    #[inline] fn next(&mut self) -> Option<uint> { self.0.next() }
    #[inline] fn size_hint(&self) -> (uint, Option<uint>) { self.0.size_hint() }
}

#[stable]
impl<'a> Iterator for Intersection<'a> {
    type Item = uint;

    #[inline] fn next(&mut self) -> Option<uint> { self.0.next() }
    #[inline] fn size_hint(&self) -> (uint, Option<uint>) { self.0.size_hint() }
}

#[stable]
impl<'a> Iterator for Difference<'a> {
    type Item = uint;

    #[inline] fn next(&mut self) -> Option<uint> { self.0.next() }
    #[inline] fn size_hint(&self) -> (uint, Option<uint>) { self.0.size_hint() }
}

#[stable]
impl<'a> Iterator for SymmetricDifference<'a> {
    type Item = uint;

    #[inline] fn next(&mut self) -> Option<uint> { self.0.next() }
    #[inline] fn size_hint(&self) -> (uint, Option<uint>) { self.0.size_hint() }
}


#[cfg(test)]
mod tests {
    use prelude::*;
    use core::u32;

    use super::Bitv;

    #[test]
    fn test_to_str() {
        let zerolen = Bitv::new();
        assert_eq!(format!("{:?}", zerolen), "");

        let eightbits = Bitv::from_elem(8u, false);
        assert_eq!(format!("{:?}", eightbits), "00000000")
    }

    #[test]
    fn test_0_elements() {
        let act = Bitv::new();
        let exp = Vec::new();
        assert!(act.eq_vec(exp.as_slice()));
        assert!(act.none() && act.all());
    }

    #[test]
    fn test_1_element() {
        let mut act = Bitv::from_elem(1u, false);
        assert!(act.eq_vec(&[false]));
        assert!(act.none() && !act.all());
        act = Bitv::from_elem(1u, true);
        assert!(act.eq_vec(&[true]));
        assert!(!act.none() && act.all());
    }

    #[test]
    fn test_2_elements() {
        let mut b = Bitv::from_elem(2, false);
        b.set(0, true);
        b.set(1, false);
        assert_eq!(format!("{:?}", b), "10");
        assert!(!b.none() && !b.all());
    }

    #[test]
    fn test_10_elements() {
        let mut act;
        // all 0

        act = Bitv::from_elem(10u, false);
        assert!((act.eq_vec(
                    &[false, false, false, false, false, false, false, false, false, false])));
        assert!(act.none() && !act.all());
        // all 1

        act = Bitv::from_elem(10u, true);
        assert!((act.eq_vec(&[true, true, true, true, true, true, true, true, true, true])));
        assert!(!act.none() && act.all());
        // mixed

        act = Bitv::from_elem(10u, false);
        act.set(0u, true);
        act.set(1u, true);
        act.set(2u, true);
        act.set(3u, true);
        act.set(4u, true);
        assert!((act.eq_vec(&[true, true, true, true, true, false, false, false, false, false])));
        assert!(!act.none() && !act.all());
        // mixed

        act = Bitv::from_elem(10u, false);
        act.set(5u, true);
        act.set(6u, true);
        act.set(7u, true);
        act.set(8u, true);
        act.set(9u, true);
        assert!((act.eq_vec(&[false, false, false, false, false, true, true, true, true, true])));
        assert!(!act.none() && !act.all());
        // mixed

        act = Bitv::from_elem(10u, false);
        act.set(0u, true);
        act.set(3u, true);
        act.set(6u, true);
        act.set(9u, true);
        assert!((act.eq_vec(&[true, false, false, true, false, false, true, false, false, true])));
        assert!(!act.none() && !act.all());
    }

    #[test]
    fn test_31_elements() {
        let mut act;
        // all 0

        act = Bitv::from_elem(31u, false);
        assert!(act.eq_vec(
                &[false, false, false, false, false, false, false, false, false, false, false,
                  false, false, false, false, false, false, false, false, false, false, false,
                  false, false, false, false, false, false, false, false, false]));
        assert!(act.none() && !act.all());
        // all 1

        act = Bitv::from_elem(31u, true);
        assert!(act.eq_vec(
                &[true, true, true, true, true, true, true, true, true, true, true, true, true,
                  true, true, true, true, true, true, true, true, true, true, true, true, true,
                  true, true, true, true, true]));
        assert!(!act.none() && act.all());
        // mixed

        act = Bitv::from_elem(31u, false);
        act.set(0u, true);
        act.set(1u, true);
        act.set(2u, true);
        act.set(3u, true);
        act.set(4u, true);
        act.set(5u, true);
        act.set(6u, true);
        act.set(7u, true);
        assert!(act.eq_vec(
                &[true, true, true, true, true, true, true, true, false, false, false, false, false,
                  false, false, false, false, false, false, false, false, false, false, false,
                  false, false, false, false, false, false, false]));
        assert!(!act.none() && !act.all());
        // mixed

        act = Bitv::from_elem(31u, false);
        act.set(16u, true);
        act.set(17u, true);
        act.set(18u, true);
        act.set(19u, true);
        act.set(20u, true);
        act.set(21u, true);
        act.set(22u, true);
        act.set(23u, true);
        assert!(act.eq_vec(
                &[false, false, false, false, false, false, false, false, false, false, false,
                  false, false, false, false, false, true, true, true, true, true, true, true, true,
                  false, false, false, false, false, false, false]));
        assert!(!act.none() && !act.all());
        // mixed

        act = Bitv::from_elem(31u, false);
        act.set(24u, true);
        act.set(25u, true);
        act.set(26u, true);
        act.set(27u, true);
        act.set(28u, true);
        act.set(29u, true);
        act.set(30u, true);
        assert!(act.eq_vec(
                &[false, false, false, false, false, false, false, false, false, false, false,
                  false, false, false, false, false, false, false, false, false, false, false,
                  false, false, true, true, true, true, true, true, true]));
        assert!(!act.none() && !act.all());
        // mixed

        act = Bitv::from_elem(31u, false);
        act.set(3u, true);
        act.set(17u, true);
        act.set(30u, true);
        assert!(act.eq_vec(
                &[false, false, false, true, false, false, false, false, false, false, false, false,
                  false, false, false, false, false, true, false, false, false, false, false, false,
                  false, false, false, false, false, false, true]));
        assert!(!act.none() && !act.all());
    }

    #[test]
    fn test_32_elements() {
        let mut act;
        // all 0

        act = Bitv::from_elem(32u, false);
        assert!(act.eq_vec(
                &[false, false, false, false, false, false, false, false, false, false, false,
                  false, false, false, false, false, false, false, false, false, false, false,
                  false, false, false, false, false, false, false, false, false, false]));
        assert!(act.none() && !act.all());
        // all 1

        act = Bitv::from_elem(32u, true);
        assert!(act.eq_vec(
                &[true, true, true, true, true, true, true, true, true, true, true, true, true,
                  true, true, true, true, true, true, true, true, true, true, true, true, true,
                  true, true, true, true, true, true]));
        assert!(!act.none() && act.all());
        // mixed

        act = Bitv::from_elem(32u, false);
        act.set(0u, true);
        act.set(1u, true);
        act.set(2u, true);
        act.set(3u, true);
        act.set(4u, true);
        act.set(5u, true);
        act.set(6u, true);
        act.set(7u, true);
        assert!(act.eq_vec(
                &[true, true, true, true, true, true, true, true, false, false, false, false, false,
                  false, false, false, false, false, false, false, false, false, false, false,
                  false, false, false, false, false, false, false, false]));
        assert!(!act.none() && !act.all());
        // mixed

        act = Bitv::from_elem(32u, false);
        act.set(16u, true);
        act.set(17u, true);
        act.set(18u, true);
        act.set(19u, true);
        act.set(20u, true);
        act.set(21u, true);
        act.set(22u, true);
        act.set(23u, true);
        assert!(act.eq_vec(
                &[false, false, false, false, false, false, false, false, false, false, false,
                  false, false, false, false, false, true, true, true, true, true, true, true, true,
                  false, false, false, false, false, false, false, false]));
        assert!(!act.none() && !act.all());
        // mixed

        act = Bitv::from_elem(32u, false);
        act.set(24u, true);
        act.set(25u, true);
        act.set(26u, true);
        act.set(27u, true);
        act.set(28u, true);
        act.set(29u, true);
        act.set(30u, true);
        act.set(31u, true);
        assert!(act.eq_vec(
                &[false, false, false, false, false, false, false, false, false, false, false,
                  false, false, false, false, false, false, false, false, false, false, false,
                  false, false, true, true, true, true, true, true, true, true]));
        assert!(!act.none() && !act.all());
        // mixed

        act = Bitv::from_elem(32u, false);
        act.set(3u, true);
        act.set(17u, true);
        act.set(30u, true);
        act.set(31u, true);
        assert!(act.eq_vec(
                &[false, false, false, true, false, false, false, false, false, false, false, false,
                  false, false, false, false, false, true, false, false, false, false, false, false,
                  false, false, false, false, false, false, true, true]));
        assert!(!act.none() && !act.all());
    }

    #[test]
    fn test_33_elements() {
        let mut act;
        // all 0

        act = Bitv::from_elem(33u, false);
        assert!(act.eq_vec(
                &[false, false, false, false, false, false, false, false, false, false, false,
                  false, false, false, false, false, false, false, false, false, false, false,
                  false, false, false, false, false, false, false, false, false, false, false]));
        assert!(act.none() && !act.all());
        // all 1

        act = Bitv::from_elem(33u, true);
        assert!(act.eq_vec(
                &[true, true, true, true, true, true, true, true, true, true, true, true, true,
                  true, true, true, true, true, true, true, true, true, true, true, true, true,
                  true, true, true, true, true, true, true]));
        assert!(!act.none() && act.all());
        // mixed

        act = Bitv::from_elem(33u, false);
        act.set(0u, true);
        act.set(1u, true);
        act.set(2u, true);
        act.set(3u, true);
        act.set(4u, true);
        act.set(5u, true);
        act.set(6u, true);
        act.set(7u, true);
        assert!(act.eq_vec(
                &[true, true, true, true, true, true, true, true, false, false, false, false, false,
                  false, false, false, false, false, false, false, false, false, false, false,
                  false, false, false, false, false, false, false, false, false]));
        assert!(!act.none() && !act.all());
        // mixed

        act = Bitv::from_elem(33u, false);
        act.set(16u, true);
        act.set(17u, true);
        act.set(18u, true);
        act.set(19u, true);
        act.set(20u, true);
        act.set(21u, true);
        act.set(22u, true);
        act.set(23u, true);
        assert!(act.eq_vec(
                &[false, false, false, false, false, false, false, false, false, false, false,
                  false, false, false, false, false, true, true, true, true, true, true, true, true,
                  false, false, false, false, false, false, false, false, false]));
        assert!(!act.none() && !act.all());
        // mixed

        act = Bitv::from_elem(33u, false);
        act.set(24u, true);
        act.set(25u, true);
        act.set(26u, true);
        act.set(27u, true);
        act.set(28u, true);
        act.set(29u, true);
        act.set(30u, true);
        act.set(31u, true);
        assert!(act.eq_vec(
                &[false, false, false, false, false, false, false, false, false, false, false,
                  false, false, false, false, false, false, false, false, false, false, false,
                  false, false, true, true, true, true, true, true, true, true, false]));
        assert!(!act.none() && !act.all());
        // mixed

        act = Bitv::from_elem(33u, false);
        act.set(3u, true);
        act.set(17u, true);
        act.set(30u, true);
        act.set(31u, true);
        act.set(32u, true);
        assert!(act.eq_vec(
                &[false, false, false, true, false, false, false, false, false, false, false, false,
                  false, false, false, false, false, true, false, false, false, false, false, false,
                  false, false, false, false, false, false, true, true, true]));
        assert!(!act.none() && !act.all());
    }

    #[test]
    fn test_equal_differing_sizes() {
        let v0 = Bitv::from_elem(10u, false);
        let v1 = Bitv::from_elem(11u, false);
        assert!(v0 != v1);
    }

    #[test]
    fn test_equal_greatly_differing_sizes() {
        let v0 = Bitv::from_elem(10u, false);
        let v1 = Bitv::from_elem(110u, false);
        assert!(v0 != v1);
    }

    #[test]
    fn test_equal_sneaky_small() {
        let mut a = Bitv::from_elem(1, false);
        a.set(0, true);

        let mut b = Bitv::from_elem(1, true);
        b.set(0, true);

        assert_eq!(a, b);
    }

    #[test]
    fn test_equal_sneaky_big() {
        let mut a = Bitv::from_elem(100, false);
        for i in range(0u, 100) {
            a.set(i, true);
        }

        let mut b = Bitv::from_elem(100, true);
        for i in range(0u, 100) {
            b.set(i, true);
        }

        assert_eq!(a, b);
    }

    #[test]
    fn test_from_bytes() {
        let bitv = Bitv::from_bytes(&[0b10110110, 0b00000000, 0b11111111]);
        let str = concat!("10110110", "00000000", "11111111");
        assert_eq!(format!("{:?}", bitv), str);
    }

    #[test]
    fn test_to_bytes() {
        let mut bv = Bitv::from_elem(3, true);
        bv.set(1, false);
        assert_eq!(bv.to_bytes(), vec!(0b10100000));

        let mut bv = Bitv::from_elem(9, false);
        bv.set(2, true);
        bv.set(8, true);
        assert_eq!(bv.to_bytes(), vec!(0b00100000, 0b10000000));
    }

    #[test]
    fn test_from_bools() {
        let bools = vec![true, false, true, true];
        let bitv: Bitv = bools.iter().map(|n| *n).collect();
        assert_eq!(format!("{:?}", bitv), "1011");
    }

    #[test]
    fn test_to_bools() {
        let bools = vec!(false, false, true, false, false, true, true, false);
        assert_eq!(Bitv::from_bytes(&[0b00100110]).iter().collect::<Vec<bool>>(), bools);
    }

    #[test]
    fn test_bitv_iterator() {
        let bools = vec![true, false, true, true];
        let bitv: Bitv = bools.iter().map(|n| *n).collect();

        assert_eq!(bitv.iter().collect::<Vec<bool>>(), bools);

        let long = range(0, 10000).map(|i| i % 2 == 0).collect::<Vec<_>>();
        let bitv: Bitv = long.iter().map(|n| *n).collect();
        assert_eq!(bitv.iter().collect::<Vec<bool>>(), long)
    }

    #[test]
    fn test_small_difference() {
        let mut b1 = Bitv::from_elem(3, false);
        let mut b2 = Bitv::from_elem(3, false);
        b1.set(0, true);
        b1.set(1, true);
        b2.set(1, true);
        b2.set(2, true);
        assert!(b1.difference(&b2));
        assert!(b1[0]);
        assert!(!b1[1]);
        assert!(!b1[2]);
    }

    #[test]
    fn test_big_difference() {
        let mut b1 = Bitv::from_elem(100, false);
        let mut b2 = Bitv::from_elem(100, false);
        b1.set(0, true);
        b1.set(40, true);
        b2.set(40, true);
        b2.set(80, true);
        assert!(b1.difference(&b2));
        assert!(b1[0]);
        assert!(!b1[40]);
        assert!(!b1[80]);
    }

    #[test]
    fn test_small_clear() {
        let mut b = Bitv::from_elem(14, true);
        assert!(!b.none() && b.all());
        b.clear();
        assert!(b.none() && !b.all());
    }

    #[test]
    fn test_big_clear() {
        let mut b = Bitv::from_elem(140, true);
        assert!(!b.none() && b.all());
        b.clear();
        assert!(b.none() && !b.all());
    }

    #[test]
    fn test_bitv_lt() {
        let mut a = Bitv::from_elem(5u, false);
        let mut b = Bitv::from_elem(5u, false);

        assert!(!(a < b) && !(b < a));
        b.set(2, true);
        assert!(a < b);
        a.set(3, true);
        assert!(a < b);
        a.set(2, true);
        assert!(!(a < b) && b < a);
        b.set(0, true);
        assert!(a < b);
    }

    #[test]
    fn test_ord() {
        let mut a = Bitv::from_elem(5u, false);
        let mut b = Bitv::from_elem(5u, false);

        assert!(a <= b && a >= b);
        a.set(1, true);
        assert!(a > b && a >= b);
        assert!(b < a && b <= a);
        b.set(1, true);
        b.set(2, true);
        assert!(b > a && b >= a);
        assert!(a < b && a <= b);
    }


    #[test]
    fn test_small_bitv_tests() {
        let v = Bitv::from_bytes(&[0]);
        assert!(!v.all());
        assert!(!v.any());
        assert!(v.none());

        let v = Bitv::from_bytes(&[0b00010100]);
        assert!(!v.all());
        assert!(v.any());
        assert!(!v.none());

        let v = Bitv::from_bytes(&[0xFF]);
        assert!(v.all());
        assert!(v.any());
        assert!(!v.none());
    }

    #[test]
    fn test_big_bitv_tests() {
        let v = Bitv::from_bytes(&[ // 88 bits
            0, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0]);
        assert!(!v.all());
        assert!(!v.any());
        assert!(v.none());

        let v = Bitv::from_bytes(&[ // 88 bits
            0, 0, 0b00010100, 0,
            0, 0, 0, 0b00110100,
            0, 0, 0]);
        assert!(!v.all());
        assert!(v.any());
        assert!(!v.none());

        let v = Bitv::from_bytes(&[ // 88 bits
            0xFF, 0xFF, 0xFF, 0xFF,
            0xFF, 0xFF, 0xFF, 0xFF,
            0xFF, 0xFF, 0xFF]);
        assert!(v.all());
        assert!(v.any());
        assert!(!v.none());
    }

    #[test]
    fn test_bitv_push_pop() {
        let mut s = Bitv::from_elem(5 * u32::BITS - 2, false);
        assert_eq!(s.len(), 5 * u32::BITS - 2);
        assert_eq!(s[5 * u32::BITS - 3], false);
        s.push(true);
        s.push(true);
        assert_eq!(s[5 * u32::BITS - 2], true);
        assert_eq!(s[5 * u32::BITS - 1], true);
        // Here the internal vector will need to be extended
        s.push(false);
        assert_eq!(s[5 * u32::BITS], false);
        s.push(false);
        assert_eq!(s[5 * u32::BITS + 1], false);
        assert_eq!(s.len(), 5 * u32::BITS + 2);
        // Pop it all off
        assert_eq!(s.pop(), Some(false));
        assert_eq!(s.pop(), Some(false));
        assert_eq!(s.pop(), Some(true));
        assert_eq!(s.pop(), Some(true));
        assert_eq!(s.len(), 5 * u32::BITS - 2);
    }

    #[test]
    fn test_bitv_truncate() {
        let mut s = Bitv::from_elem(5 * u32::BITS, true);

        assert_eq!(s, Bitv::from_elem(5 * u32::BITS, true));
        assert_eq!(s.len(), 5 * u32::BITS);
        s.truncate(4 * u32::BITS);
        assert_eq!(s, Bitv::from_elem(4 * u32::BITS, true));
        assert_eq!(s.len(), 4 * u32::BITS);
        // Truncating to a size > s.len() should be a noop
        s.truncate(5 * u32::BITS);
        assert_eq!(s, Bitv::from_elem(4 * u32::BITS, true));
        assert_eq!(s.len(), 4 * u32::BITS);
        s.truncate(3 * u32::BITS - 10);
        assert_eq!(s, Bitv::from_elem(3 * u32::BITS - 10, true));
        assert_eq!(s.len(), 3 * u32::BITS - 10);
        s.truncate(0);
        assert_eq!(s, Bitv::from_elem(0, true));
        assert_eq!(s.len(), 0);
    }

    #[test]
    fn test_bitv_reserve() {
        let mut s = Bitv::from_elem(5 * u32::BITS, true);
        // Check capacity
        assert!(s.capacity() >= 5 * u32::BITS);
        s.reserve(2 * u32::BITS);
        assert!(s.capacity() >= 7 * u32::BITS);
        s.reserve(7 * u32::BITS);
        assert!(s.capacity() >= 12 * u32::BITS);
        s.reserve_exact(7 * u32::BITS);
        assert!(s.capacity() >= 12 * u32::BITS);
        s.reserve(7 * u32::BITS + 1);
        assert!(s.capacity() >= 12 * u32::BITS + 1);
        // Check that length hasn't changed
        assert_eq!(s.len(), 5 * u32::BITS);
        s.push(true);
        s.push(false);
        s.push(true);
        assert_eq!(s[5 * u32::BITS - 1], true);
        assert_eq!(s[5 * u32::BITS - 0], true);
        assert_eq!(s[5 * u32::BITS + 1], false);
        assert_eq!(s[5 * u32::BITS + 2], true);
    }

    #[test]
    fn test_bitv_grow() {
        let mut bitv = Bitv::from_bytes(&[0b10110110, 0b00000000, 0b10101010]);
        bitv.grow(32, true);
        assert_eq!(bitv, Bitv::from_bytes(&[0b10110110, 0b00000000, 0b10101010,
                                     0xFF, 0xFF, 0xFF, 0xFF]));
        bitv.grow(64, false);
        assert_eq!(bitv, Bitv::from_bytes(&[0b10110110, 0b00000000, 0b10101010,
                                     0xFF, 0xFF, 0xFF, 0xFF, 0, 0, 0, 0, 0, 0, 0, 0]));
        bitv.grow(16, true);
        assert_eq!(bitv, Bitv::from_bytes(&[0b10110110, 0b00000000, 0b10101010,
                                     0xFF, 0xFF, 0xFF, 0xFF, 0, 0, 0, 0, 0, 0, 0, 0, 0xFF, 0xFF]));
    }

    #[test]
    fn test_bitv_extend() {
        let mut bitv = Bitv::from_bytes(&[0b10110110, 0b00000000, 0b11111111]);
        let ext = Bitv::from_bytes(&[0b01001001, 0b10010010, 0b10111101]);
        bitv.extend(ext.iter());
        assert_eq!(bitv, Bitv::from_bytes(&[0b10110110, 0b00000000, 0b11111111,
                                     0b01001001, 0b10010010, 0b10111101]));
    }
}




#[cfg(test)]
mod bitv_bench {
    use std::prelude::v1::*;
    use std::rand;
    use std::rand::Rng;
    use std::u32;
    use test::{Bencher, black_box};

    use super::Bitv;

    static BENCH_BITS : uint = 1 << 14;

    fn rng() -> rand::IsaacRng {
        let seed: &[_] = &[1, 2, 3, 4, 5, 6, 7, 8, 9, 0];
        rand::SeedableRng::from_seed(seed)
    }

    #[bench]
    fn bench_uint_small(b: &mut Bencher) {
        let mut r = rng();
        let mut bitv = 0 as uint;
        b.iter(|| {
            for _ in range(0u, 100) {
                bitv |= 1 << ((r.next_u32() as uint) % u32::BITS);
            }
            black_box(&bitv);
        });
    }

    #[bench]
    fn bench_bitv_set_big_fixed(b: &mut Bencher) {
        let mut r = rng();
        let mut bitv = Bitv::from_elem(BENCH_BITS, false);
        b.iter(|| {
            for _ in range(0u, 100) {
                bitv.set((r.next_u32() as uint) % BENCH_BITS, true);
            }
            black_box(&bitv);
        });
    }

    #[bench]
    fn bench_bitv_set_big_variable(b: &mut Bencher) {
        let mut r = rng();
        let mut bitv = Bitv::from_elem(BENCH_BITS, false);
        b.iter(|| {
            for _ in range(0u, 100) {
                bitv.set((r.next_u32() as uint) % BENCH_BITS, r.gen());
            }
            black_box(&bitv);
        });
    }

    #[bench]
    fn bench_bitv_set_small(b: &mut Bencher) {
        let mut r = rng();
        let mut bitv = Bitv::from_elem(u32::BITS, false);
        b.iter(|| {
            for _ in range(0u, 100) {
                bitv.set((r.next_u32() as uint) % u32::BITS, true);
            }
            black_box(&bitv);
        });
    }

    #[bench]
    fn bench_bitv_big_union(b: &mut Bencher) {
        let mut b1 = Bitv::from_elem(BENCH_BITS, false);
        let b2 = Bitv::from_elem(BENCH_BITS, false);
        b.iter(|| {
            b1.union(&b2)
        })
    }

    #[bench]
    fn bench_bitv_small_iter(b: &mut Bencher) {
        let bitv = Bitv::from_elem(u32::BITS, false);
        b.iter(|| {
            let mut sum = 0u;
            for _ in range(0u, 10) {
                for pres in bitv.iter() {
                    sum += pres as uint;
                }
            }
            sum
        })
    }

    #[bench]
    fn bench_bitv_big_iter(b: &mut Bencher) {
        let bitv = Bitv::from_elem(BENCH_BITS, false);
        b.iter(|| {
            let mut sum = 0u;
            for pres in bitv.iter() {
                sum += pres as uint;
            }
            sum
        })
    }
}







#[cfg(test)]
mod bitv_set_test {
    use prelude::*;
    use std::iter::range_step;

    use super::{Bitv, BitvSet};

    #[test]
    fn test_bitv_set_show() {
        let mut s = BitvSet::new();
        s.insert(1);
        s.insert(10);
        s.insert(50);
        s.insert(2);
        assert_eq!("BitvSet {1u, 2u, 10u, 50u}", format!("{:?}", s));
    }

    #[test]
    fn test_bitv_set_from_uints() {
        let uints = vec![0, 2, 2, 3];
        let a: BitvSet = uints.into_iter().collect();
        let mut b = BitvSet::new();
        b.insert(0);
        b.insert(2);
        b.insert(3);
        assert_eq!(a, b);
    }

    #[test]
    fn test_bitv_set_iterator() {
        let uints = vec![0, 2, 2, 3];
        let bitv: BitvSet = uints.into_iter().collect();

        let idxs: Vec<uint> = bitv.iter().collect();
        assert_eq!(idxs, vec![0, 2, 3]);

        let long: BitvSet = range(0u, 10000).filter(|&n| n % 2 == 0).collect();
        let real = range_step(0, 10000, 2).collect::<Vec<uint>>();

        let idxs: Vec<uint> = long.iter().collect();
        assert_eq!(idxs, real);
    }

    #[test]
    fn test_bitv_set_frombitv_init() {
        let bools = [true, false];
        let lengths = [10, 64, 100];
        for &b in bools.iter() {
            for &l in lengths.iter() {
                let bitset = BitvSet::from_bitv(Bitv::from_elem(l, b));
                assert_eq!(bitset.contains(&1u), b);
                assert_eq!(bitset.contains(&(l-1u)), b);
                assert!(!bitset.contains(&l));
            }
        }
    }

    #[test]
    fn test_bitv_masking() {
        let b = Bitv::from_elem(140, true);
        let mut bs = BitvSet::from_bitv(b);
        assert!(bs.contains(&139));
        assert!(!bs.contains(&140));
        assert!(bs.insert(150));
        assert!(!bs.contains(&140));
        assert!(!bs.contains(&149));
        assert!(bs.contains(&150));
        assert!(!bs.contains(&151));
    }

    #[test]
    fn test_bitv_set_basic() {
        let mut b = BitvSet::new();
        assert!(b.insert(3));
        assert!(!b.insert(3));
        assert!(b.contains(&3));
        assert!(b.insert(4));
        assert!(!b.insert(4));
        assert!(b.contains(&3));
        assert!(b.insert(400));
        assert!(!b.insert(400));
        assert!(b.contains(&400));
        assert_eq!(b.len(), 3);
    }

    #[test]
    fn test_bitv_set_intersection() {
        let mut a = BitvSet::new();
        let mut b = BitvSet::new();

        assert!(a.insert(11));
        assert!(a.insert(1));
        assert!(a.insert(3));
        assert!(a.insert(77));
        assert!(a.insert(103));
        assert!(a.insert(5));

        assert!(b.insert(2));
        assert!(b.insert(11));
        assert!(b.insert(77));
        assert!(b.insert(5));
        assert!(b.insert(3));

        let expected = [3, 5, 11, 77];
        let actual = a.intersection(&b).collect::<Vec<uint>>();
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_bitv_set_difference() {
        let mut a = BitvSet::new();
        let mut b = BitvSet::new();

        assert!(a.insert(1));
        assert!(a.insert(3));
        assert!(a.insert(5));
        assert!(a.insert(200));
        assert!(a.insert(500));

        assert!(b.insert(3));
        assert!(b.insert(200));

        let expected = [1, 5, 500];
        let actual = a.difference(&b).collect::<Vec<uint>>();
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_bitv_set_symmetric_difference() {
        let mut a = BitvSet::new();
        let mut b = BitvSet::new();

        assert!(a.insert(1));
        assert!(a.insert(3));
        assert!(a.insert(5));
        assert!(a.insert(9));
        assert!(a.insert(11));

        assert!(b.insert(3));
        assert!(b.insert(9));
        assert!(b.insert(14));
        assert!(b.insert(220));

        let expected = [1, 5, 11, 14, 220];
        let actual = a.symmetric_difference(&b).collect::<Vec<uint>>();
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_bitv_set_union() {
        let mut a = BitvSet::new();
        let mut b = BitvSet::new();
        assert!(a.insert(1));
        assert!(a.insert(3));
        assert!(a.insert(5));
        assert!(a.insert(9));
        assert!(a.insert(11));
        assert!(a.insert(160));
        assert!(a.insert(19));
        assert!(a.insert(24));
        assert!(a.insert(200));

        assert!(b.insert(1));
        assert!(b.insert(5));
        assert!(b.insert(9));
        assert!(b.insert(13));
        assert!(b.insert(19));

        let expected = [1, 3, 5, 9, 11, 13, 19, 24, 160, 200];
        let actual = a.union(&b).collect::<Vec<uint>>();
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_bitv_set_subset() {
        let mut set1 = BitvSet::new();
        let mut set2 = BitvSet::new();

        assert!(set1.is_subset(&set2)); //  {}  {}
        set2.insert(100);
        assert!(set1.is_subset(&set2)); //  {}  { 1 }
        set2.insert(200);
        assert!(set1.is_subset(&set2)); //  {}  { 1, 2 }
        set1.insert(200);
        assert!(set1.is_subset(&set2)); //  { 2 }  { 1, 2 }
        set1.insert(300);
        assert!(!set1.is_subset(&set2)); // { 2, 3 }  { 1, 2 }
        set2.insert(300);
        assert!(set1.is_subset(&set2)); // { 2, 3 }  { 1, 2, 3 }
        set2.insert(400);
        assert!(set1.is_subset(&set2)); // { 2, 3 }  { 1, 2, 3, 4 }
        set2.remove(&100);
        assert!(set1.is_subset(&set2)); // { 2, 3 }  { 2, 3, 4 }
        set2.remove(&300);
        assert!(!set1.is_subset(&set2)); // { 2, 3 }  { 2, 4 }
        set1.remove(&300);
        assert!(set1.is_subset(&set2)); // { 2 }  { 2, 4 }
    }

    #[test]
    fn test_bitv_set_is_disjoint() {
        let a = BitvSet::from_bitv(Bitv::from_bytes(&[0b10100010]));
        let b = BitvSet::from_bitv(Bitv::from_bytes(&[0b01000000]));
        let c = BitvSet::new();
        let d = BitvSet::from_bitv(Bitv::from_bytes(&[0b00110000]));

        assert!(!a.is_disjoint(&d));
        assert!(!d.is_disjoint(&a));

        assert!(a.is_disjoint(&b));
        assert!(a.is_disjoint(&c));
        assert!(b.is_disjoint(&a));
        assert!(b.is_disjoint(&c));
        assert!(c.is_disjoint(&a));
        assert!(c.is_disjoint(&b));
    }

    #[test]
    fn test_bitv_set_union_with() {
        //a should grow to include larger elements
        let mut a = BitvSet::new();
        a.insert(0);
        let mut b = BitvSet::new();
        b.insert(5);
        let expected = BitvSet::from_bitv(Bitv::from_bytes(&[0b10000100]));
        a.union_with(&b);
        assert_eq!(a, expected);

        // Standard
        let mut a = BitvSet::from_bitv(Bitv::from_bytes(&[0b10100010]));
        let mut b = BitvSet::from_bitv(Bitv::from_bytes(&[0b01100010]));
        let c = a.clone();
        a.union_with(&b);
        b.union_with(&c);
        assert_eq!(a.len(), 4);
        assert_eq!(b.len(), 4);
    }

    #[test]
    fn test_bitv_set_intersect_with() {
        // Explicitly 0'ed bits
        let mut a = BitvSet::from_bitv(Bitv::from_bytes(&[0b10100010]));
        let mut b = BitvSet::from_bitv(Bitv::from_bytes(&[0b00000000]));
        let c = a.clone();
        a.intersect_with(&b);
        b.intersect_with(&c);
        assert!(a.is_empty());
        assert!(b.is_empty());

        // Uninitialized bits should behave like 0's
        let mut a = BitvSet::from_bitv(Bitv::from_bytes(&[0b10100010]));
        let mut b = BitvSet::new();
        let c = a.clone();
        a.intersect_with(&b);
        b.intersect_with(&c);
        assert!(a.is_empty());
        assert!(b.is_empty());

        // Standard
        let mut a = BitvSet::from_bitv(Bitv::from_bytes(&[0b10100010]));
        let mut b = BitvSet::from_bitv(Bitv::from_bytes(&[0b01100010]));
        let c = a.clone();
        a.intersect_with(&b);
        b.intersect_with(&c);
        assert_eq!(a.len(), 2);
        assert_eq!(b.len(), 2);
    }

    #[test]
    fn test_bitv_set_difference_with() {
        // Explicitly 0'ed bits
        let mut a = BitvSet::from_bitv(Bitv::from_bytes(&[0b00000000]));
        let b = BitvSet::from_bitv(Bitv::from_bytes(&[0b10100010]));
        a.difference_with(&b);
        assert!(a.is_empty());

        // Uninitialized bits should behave like 0's
        let mut a = BitvSet::new();
        let b = BitvSet::from_bitv(Bitv::from_bytes(&[0b11111111]));
        a.difference_with(&b);
        assert!(a.is_empty());

        // Standard
        let mut a = BitvSet::from_bitv(Bitv::from_bytes(&[0b10100010]));
        let mut b = BitvSet::from_bitv(Bitv::from_bytes(&[0b01100010]));
        let c = a.clone();
        a.difference_with(&b);
        b.difference_with(&c);
        assert_eq!(a.len(), 1);
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_bitv_set_symmetric_difference_with() {
        //a should grow to include larger elements
        let mut a = BitvSet::new();
        a.insert(0);
        a.insert(1);
        let mut b = BitvSet::new();
        b.insert(1);
        b.insert(5);
        let expected = BitvSet::from_bitv(Bitv::from_bytes(&[0b10000100]));
        a.symmetric_difference_with(&b);
        assert_eq!(a, expected);

        let mut a = BitvSet::from_bitv(Bitv::from_bytes(&[0b10100010]));
        let b = BitvSet::new();
        let c = a.clone();
        a.symmetric_difference_with(&b);
        assert_eq!(a, c);

        // Standard
        let mut a = BitvSet::from_bitv(Bitv::from_bytes(&[0b11100010]));
        let mut b = BitvSet::from_bitv(Bitv::from_bytes(&[0b01101010]));
        let c = a.clone();
        a.symmetric_difference_with(&b);
        b.symmetric_difference_with(&c);
        assert_eq!(a.len(), 2);
        assert_eq!(b.len(), 2);
    }

    #[test]
    fn test_bitv_set_eq() {
        let a = BitvSet::from_bitv(Bitv::from_bytes(&[0b10100010]));
        let b = BitvSet::from_bitv(Bitv::from_bytes(&[0b00000000]));
        let c = BitvSet::new();

        assert!(a == a);
        assert!(a != b);
        assert!(a != c);
        assert!(b == b);
        assert!(b == c);
        assert!(c == c);
    }

    #[test]
    fn test_bitv_set_cmp() {
        let a = BitvSet::from_bitv(Bitv::from_bytes(&[0b10100010]));
        let b = BitvSet::from_bitv(Bitv::from_bytes(&[0b00000000]));
        let c = BitvSet::new();

        assert_eq!(a.cmp(&b), Greater);
        assert_eq!(a.cmp(&c), Greater);
        assert_eq!(b.cmp(&a), Less);
        assert_eq!(b.cmp(&c), Equal);
        assert_eq!(c.cmp(&a), Less);
        assert_eq!(c.cmp(&b), Equal);
    }

    #[test]
    fn test_bitv_remove() {
        let mut a = BitvSet::new();

        assert!(a.insert(1));
        assert!(a.remove(&1));

        assert!(a.insert(100));
        assert!(a.remove(&100));

        assert!(a.insert(1000));
        assert!(a.remove(&1000));
        a.shrink_to_fit();
    }

    #[test]
    fn test_bitv_clone() {
        let mut a = BitvSet::new();

        assert!(a.insert(1));
        assert!(a.insert(100));
        assert!(a.insert(1000));

        let mut b = a.clone();

        assert!(a == b);

        assert!(b.remove(&1));
        assert!(a.contains(&1));

        assert!(a.remove(&1000));
        assert!(b.contains(&1000));
    }
}





#[cfg(test)]
mod bitv_set_bench {
    use std::prelude::v1::*;
    use std::rand;
    use std::rand::Rng;
    use std::u32;
    use test::{Bencher, black_box};

    use super::{Bitv, BitvSet};

    static BENCH_BITS : uint = 1 << 14;

    fn rng() -> rand::IsaacRng {
        let seed: &[_] = &[1, 2, 3, 4, 5, 6, 7, 8, 9, 0];
        rand::SeedableRng::from_seed(seed)
    }

    #[bench]
    fn bench_bitvset_small(b: &mut Bencher) {
        let mut r = rng();
        let mut bitv = BitvSet::new();
        b.iter(|| {
            for _ in range(0u, 100) {
                bitv.insert((r.next_u32() as uint) % u32::BITS);
            }
            black_box(&bitv);
        });
    }

    #[bench]
    fn bench_bitvset_big(b: &mut Bencher) {
        let mut r = rng();
        let mut bitv = BitvSet::new();
        b.iter(|| {
            for _ in range(0u, 100) {
                bitv.insert((r.next_u32() as uint) % BENCH_BITS);
            }
            black_box(&bitv);
        });
    }

    #[bench]
    fn bench_bitvset_iter(b: &mut Bencher) {
        let bitv = BitvSet::from_bitv(Bitv::from_fn(BENCH_BITS,
                                              |idx| {idx % 3 == 0}));
        b.iter(|| {
            let mut sum = 0u;
            for idx in bitv.iter() {
                sum += idx as uint;
            }
            sum
        })
    }
}
