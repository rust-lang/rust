// Copyright 2013-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Interface to random number generators in Rust.
//!
//! This is an experimental library which lives underneath the standard library
//! in its dependency chain. This library is intended to define the interface
//! for random number generation and also provide utilities around doing so. It
//! is not recommended to use this library directly, but rather the official
//! interface through `std::rand`.

#![crate_name = "rand"]
#![crate_type = "rlib"]
#![doc(html_logo_url = "https://www.rust-lang.org/logos/rust-logo-128x128-blk.png",
       html_favicon_url = "https://doc.rust-lang.org/favicon.ico",
       html_root_url = "https://doc.rust-lang.org/nightly/",
       html_playground_url = "https://play.rust-lang.org/",
       test(attr(deny(warnings))))]
#![deny(warnings)]
#![deny(missing_debug_implementations)]
#![no_std]
#![unstable(feature = "rand",
            reason = "use `rand` from crates.io",
            issue = "27703")]
#![feature(core_intrinsics)]
#![feature(staged_api)]
#![feature(step_by)]
#![feature(custom_attribute)]
#![feature(specialization)]
#![allow(unused_attributes)]

#![cfg_attr(not(test), feature(core_float))] // only necessary for no_std
#![cfg_attr(test, feature(test, rand))]

#![allow(deprecated)]

#[cfg(test)]
#[macro_use]
extern crate std;

use core::fmt;
use core::f64;
use core::intrinsics;
use core::marker::PhantomData;

pub use isaac::{Isaac64Rng, IsaacRng};
pub use chacha::ChaChaRng;

use distributions::{IndependentSample, Range};
use distributions::range::SampleRange;

#[cfg(test)]
const RAND_BENCH_N: u64 = 100;

pub mod distributions;
pub mod isaac;
pub mod chacha;
pub mod reseeding;
mod rand_impls;

// Temporary trait to implement a few floating-point routines
// needed by librand; this is necessary because librand doesn't
// depend on libstd.  This will go away when librand is integrated
// into libstd.
#[doc(hidden)]
trait FloatMath: Sized {
    fn exp(self) -> Self;
    fn ln(self) -> Self;
    fn sqrt(self) -> Self;
    fn powf(self, n: Self) -> Self;
}

impl FloatMath for f64 {
    #[inline]
    fn exp(self) -> f64 {
        unsafe { intrinsics::expf64(self) }
    }

    #[inline]
    fn ln(self) -> f64 {
        unsafe { intrinsics::logf64(self) }
    }

    #[inline]
    fn powf(self, n: f64) -> f64 {
        unsafe { intrinsics::powf64(self, n) }
    }

    #[inline]
    fn sqrt(self) -> f64 {
        if self < 0.0 {
            f64::NAN
        } else {
            unsafe { intrinsics::sqrtf64(self) }
        }
    }
}

/// A type that can be randomly generated using an `Rng`.
#[doc(hidden)]
pub trait Rand: Sized {
    /// Generates a random instance of this type using the specified source of
    /// randomness.
    fn rand<R: Rng>(rng: &mut R) -> Self;
}

/// A random number generator.
pub trait Rng: Sized {
    /// Return the next random u32.
    ///
    /// This rarely needs to be called directly, prefer `r.gen()` to
    /// `r.next_u32()`.
    // FIXME #7771: Should be implemented in terms of next_u64
    fn next_u32(&mut self) -> u32;

    /// Return the next random u64.
    ///
    /// By default this is implemented in terms of `next_u32`. An
    /// implementation of this trait must provide at least one of
    /// these two methods. Similarly to `next_u32`, this rarely needs
    /// to be called directly, prefer `r.gen()` to `r.next_u64()`.
    fn next_u64(&mut self) -> u64 {
        ((self.next_u32() as u64) << 32) | (self.next_u32() as u64)
    }

    /// Return the next random f32 selected from the half-open
    /// interval `[0, 1)`.
    ///
    /// By default this is implemented in terms of `next_u32`, but a
    /// random number generator which can generate numbers satisfying
    /// the requirements directly can overload this for performance.
    /// It is required that the return value lies in `[0, 1)`.
    ///
    /// See `Closed01` for the closed interval `[0,1]`, and
    /// `Open01` for the open interval `(0,1)`.
    fn next_f32(&mut self) -> f32 {
        const MANTISSA_BITS: usize = 24;
        const IGNORED_BITS: usize = 8;
        const SCALE: f32 = (1u64 << MANTISSA_BITS) as f32;

        // using any more than `MANTISSA_BITS` bits will
        // cause (e.g.) 0xffff_ffff to correspond to 1
        // exactly, so we need to drop some (8 for f32, 11
        // for f64) to guarantee the open end.
        (self.next_u32() >> IGNORED_BITS) as f32 / SCALE
    }

    /// Return the next random f64 selected from the half-open
    /// interval `[0, 1)`.
    ///
    /// By default this is implemented in terms of `next_u64`, but a
    /// random number generator which can generate numbers satisfying
    /// the requirements directly can overload this for performance.
    /// It is required that the return value lies in `[0, 1)`.
    ///
    /// See `Closed01` for the closed interval `[0,1]`, and
    /// `Open01` for the open interval `(0,1)`.
    fn next_f64(&mut self) -> f64 {
        const MANTISSA_BITS: usize = 53;
        const IGNORED_BITS: usize = 11;
        const SCALE: f64 = (1u64 << MANTISSA_BITS) as f64;

        (self.next_u64() >> IGNORED_BITS) as f64 / SCALE
    }

    /// Fill `dest` with random data.
    ///
    /// This has a default implementation in terms of `next_u64` and
    /// `next_u32`, but should be overridden by implementations that
    /// offer a more efficient solution than just calling those
    /// methods repeatedly.
    ///
    /// This method does *not* have a requirement to bear any fixed
    /// relationship to the other methods, for example, it does *not*
    /// have to result in the same output as progressively filling
    /// `dest` with `self.gen::<u8>()`, and any such behaviour should
    /// not be relied upon.
    ///
    /// This method should guarantee that `dest` is entirely filled
    /// with new data, and may panic if this is impossible
    /// (e.g. reading past the end of a file that is being used as the
    /// source of randomness).
    fn fill_bytes(&mut self, dest: &mut [u8]) {
        // this could, in theory, be done by transmuting dest to a
        // [u64], but this is (1) likely to be undefined behaviour for
        // LLVM, (2) has to be very careful about alignment concerns,
        // (3) adds more `unsafe` that needs to be checked, (4)
        // probably doesn't give much performance gain if
        // optimisations are on.
        let mut count = 0;
        let mut num = 0;
        for byte in dest {
            if count == 0 {
                // we could micro-optimise here by generating a u32 if
                // we only need a few more bytes to fill the vector
                // (i.e. at most 4).
                num = self.next_u64();
                count = 8;
            }

            *byte = (num & 0xff) as u8;
            num >>= 8;
            count -= 1;
        }
    }

    /// Return a random value of a `Rand` type.
    #[inline(always)]
    fn gen<T: Rand>(&mut self) -> T {
        Rand::rand(self)
    }

    /// Return an iterator that will yield an infinite number of randomly
    /// generated items.
    fn gen_iter<'a, T: Rand>(&'a mut self) -> Generator<'a, T, Self> {
        Generator {
            rng: self,
            _marker: PhantomData,
        }
    }

    /// Generate a random value in the range [`low`, `high`).
    ///
    /// This is a convenience wrapper around
    /// `distributions::Range`. If this function will be called
    /// repeatedly with the same arguments, one should use `Range`, as
    /// that will amortize the computations that allow for perfect
    /// uniformity, as they only happen on initialization.
    ///
    /// # Panics
    ///
    /// Panics if `low >= high`.
    fn gen_range<T: PartialOrd + SampleRange>(&mut self, low: T, high: T) -> T {
        assert!(low < high, "Rng.gen_range called with low >= high");
        Range::new(low, high).ind_sample(self)
    }

    /// Return a bool with a 1 in n chance of true
    fn gen_weighted_bool(&mut self, n: usize) -> bool {
        n <= 1 || self.gen_range(0, n) == 0
    }

    /// Return an iterator of random characters from the set A-Z,a-z,0-9.
    fn gen_ascii_chars<'a>(&'a mut self) -> AsciiGenerator<'a, Self> {
        AsciiGenerator { rng: self }
    }

    /// Return a random element from `values`.
    ///
    /// Return `None` if `values` is empty.
    fn choose<'a, T>(&mut self, values: &'a [T]) -> Option<&'a T> {
        if values.is_empty() {
            None
        } else {
            Some(&values[self.gen_range(0, values.len())])
        }
    }

    /// Shuffle a mutable slice in place.
    fn shuffle<T>(&mut self, values: &mut [T]) {
        let mut i = values.len();
        while i >= 2 {
            // invariant: elements with index >= i have been locked in place.
            i -= 1;
            // lock element i in place.
            values.swap(i, self.gen_range(0, i + 1));
        }
    }
}

/// Iterator which will generate a stream of random items.
///
/// This iterator is created via the `gen_iter` method on `Rng`.
pub struct Generator<'a, T, R: 'a> {
    rng: &'a mut R,
    _marker: PhantomData<T>,
}

impl<'a, T: Rand, R: Rng> Iterator for Generator<'a, T, R> {
    type Item = T;

    fn next(&mut self) -> Option<T> {
        Some(self.rng.gen())
    }
}

impl<'a, T, R: fmt::Debug> fmt::Debug for Generator<'a, T, R> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("Generator")
         .field("rng", &self.rng)
         .finish()
    }
}

/// Iterator which will continuously generate random ascii characters.
///
/// This iterator is created via the `gen_ascii_chars` method on `Rng`.
pub struct AsciiGenerator<'a, R: 'a> {
    rng: &'a mut R,
}

impl<'a, R: Rng> Iterator for AsciiGenerator<'a, R> {
    type Item = char;

    fn next(&mut self) -> Option<char> {
        const GEN_ASCII_STR_CHARSET: &'static [u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZ\
                                                       abcdefghijklmnopqrstuvwxyz\
                                                       0123456789";
        Some(*self.rng.choose(GEN_ASCII_STR_CHARSET).unwrap() as char)
    }
}

impl<'a, R: fmt::Debug> fmt::Debug for AsciiGenerator<'a, R> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("AsciiGenerator")
         .field("rng", &self.rng)
         .finish()
    }
}

/// A random number generator that can be explicitly seeded to produce
/// the same stream of randomness multiple times.
pub trait SeedableRng<Seed>: Rng {
    /// Reseed an RNG with the given seed.
    fn reseed(&mut self, Seed);

    /// Create a new RNG with the given seed.
    fn from_seed(seed: Seed) -> Self;
}

/// An Xorshift[1] random number
/// generator.
///
/// The Xorshift algorithm is not suitable for cryptographic purposes
/// but is very fast. If you do not know for sure that it fits your
/// requirements, use a more secure one such as `IsaacRng` or `OsRng`.
///
/// [1]: Marsaglia, George (July 2003). ["Xorshift
/// RNGs"](http://www.jstatsoft.org/v08/i14/paper). *Journal of
/// Statistical Software*. Vol. 8 (Issue 14).
#[derive(Clone, Debug)]
pub struct XorShiftRng {
    x: u32,
    y: u32,
    z: u32,
    w: u32,
}

impl XorShiftRng {
    /// Creates a new XorShiftRng instance which is not seeded.
    ///
    /// The initial values of this RNG are constants, so all generators created
    /// by this function will yield the same stream of random numbers. It is
    /// highly recommended that this is created through `SeedableRng` instead of
    /// this function
    pub fn new_unseeded() -> XorShiftRng {
        XorShiftRng {
            x: 0x193a6754,
            y: 0xa8a7d469,
            z: 0x97830e05,
            w: 0x113ba7bb,
        }
    }
}

impl Rng for XorShiftRng {
    #[inline]
    fn next_u32(&mut self) -> u32 {
        let x = self.x;
        let t = x ^ (x << 11);
        self.x = self.y;
        self.y = self.z;
        self.z = self.w;
        let w = self.w;
        self.w = w ^ (w >> 19) ^ (t ^ (t >> 8));
        self.w
    }
}

impl SeedableRng<[u32; 4]> for XorShiftRng {
    /// Reseed an XorShiftRng. This will panic if `seed` is entirely 0.
    fn reseed(&mut self, seed: [u32; 4]) {
        assert!(!seed.iter().all(|&x| x == 0),
                "XorShiftRng.reseed called with an all zero seed.");

        self.x = seed[0];
        self.y = seed[1];
        self.z = seed[2];
        self.w = seed[3];
    }

    /// Create a new XorShiftRng. This will panic if `seed` is entirely 0.
    fn from_seed(seed: [u32; 4]) -> XorShiftRng {
        assert!(!seed.iter().all(|&x| x == 0),
                "XorShiftRng::from_seed called with an all zero seed.");

        XorShiftRng {
            x: seed[0],
            y: seed[1],
            z: seed[2],
            w: seed[3],
        }
    }
}

impl Rand for XorShiftRng {
    fn rand<R: Rng>(rng: &mut R) -> XorShiftRng {
        let mut tuple: (u32, u32, u32, u32) = rng.gen();
        while tuple == (0, 0, 0, 0) {
            tuple = rng.gen();
        }
        let (x, y, z, w) = tuple;
        XorShiftRng {
            x: x,
            y: y,
            z: z,
            w: w,
        }
    }
}

/// A wrapper for generating floating point numbers uniformly in the
/// open interval `(0,1)` (not including either endpoint).
///
/// Use `Closed01` for the closed interval `[0,1]`, and the default
/// `Rand` implementation for `f32` and `f64` for the half-open
/// `[0,1)`.
pub struct Open01<F>(pub F);

impl<F: fmt::Debug> fmt::Debug for Open01<F> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_tuple("Open01")
         .field(&self.0)
         .finish()
    }
}

/// A wrapper for generating floating point numbers uniformly in the
/// closed interval `[0,1]` (including both endpoints).
///
/// Use `Open01` for the closed interval `(0,1)`, and the default
/// `Rand` implementation of `f32` and `f64` for the half-open
/// `[0,1)`.
pub struct Closed01<F>(pub F);

impl<F: fmt::Debug> fmt::Debug for Closed01<F> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_tuple("Closed01")
         .field(&self.0)
         .finish()
    }
}

#[cfg(test)]
mod test {
    use std::__rand as rand;

    pub struct MyRng<R> {
        inner: R,
    }

    impl<R: rand::Rng> ::Rng for MyRng<R> {
        fn next_u32(&mut self) -> u32 {
            rand::Rng::next_u32(&mut self.inner)
        }
    }

    pub fn rng() -> MyRng<rand::ThreadRng> {
        MyRng { inner: rand::thread_rng() }
    }

    pub fn weak_rng() -> MyRng<rand::ThreadRng> {
        MyRng { inner: rand::thread_rng() }
    }
}
