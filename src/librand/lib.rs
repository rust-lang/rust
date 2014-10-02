// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
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
#![license = "MIT/ASL2"]
#![crate_type = "rlib"]
#![doc(html_logo_url = "http://www.rust-lang.org/logos/rust-logo-128x128-blk.png",
       html_favicon_url = "http://www.rust-lang.org/favicon.ico",
       html_root_url = "http://doc.rust-lang.org/master/",
       html_playground_url = "http://play.rust-lang.org/")]

#![feature(macro_rules, phase, globs)]
#![no_std]
#![experimental]

#[phase(plugin, link)]
extern crate core;

#[cfg(test)] #[phase(plugin, link)] extern crate std;
#[cfg(test)] #[phase(plugin, link)] extern crate log;
#[cfg(test)] extern crate native;
#[cfg(test)] extern crate debug;

use core::prelude::*;

pub use isaac::{IsaacRng, Isaac64Rng};

use distributions::{Range, IndependentSample};
use distributions::range::SampleRange;

#[cfg(test)]
static RAND_BENCH_N: u64 = 100;

pub mod distributions;
pub mod isaac;
pub mod reseeding;
mod rand_impls;

/// A type that can be randomly generated using an `Rng`.
pub trait Rand {
    /// Generates a random instance of this type using the specified source of
    /// randomness.
    fn rand<R: Rng>(rng: &mut R) -> Self;
}

/// A random number generator.
pub trait Rng {
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
        (self.next_u32() as u64 << 32) | (self.next_u32() as u64)
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
    /// with new data, and may fail if this is impossible
    /// (e.g. reading past the end of a file that is being used as the
    /// source of randomness).
    ///
    /// # Example
    ///
    /// ```rust
    /// use std::rand::{task_rng, Rng};
    ///
    /// let mut v = [0u8, .. 13579];
    /// task_rng().fill_bytes(v);
    /// println!("{}", v.as_slice());
    /// ```
    fn fill_bytes(&mut self, dest: &mut [u8]) {
        // this could, in theory, be done by transmuting dest to a
        // [u64], but this is (1) likely to be undefined behaviour for
        // LLVM, (2) has to be very careful about alignment concerns,
        // (3) adds more `unsafe` that needs to be checked, (4)
        // probably doesn't give much performance gain if
        // optimisations are on.
        let mut count = 0i;
        let mut num = 0;
        for byte in dest.iter_mut() {
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
    ///
    /// # Example
    ///
    /// ```rust
    /// use std::rand::{task_rng, Rng};
    ///
    /// let mut rng = task_rng();
    /// let x: uint = rng.gen();
    /// println!("{}", x);
    /// println!("{}", rng.gen::<(f64, bool)>());
    /// ```
    #[inline(always)]
    fn gen<T: Rand>(&mut self) -> T {
        Rand::rand(self)
    }

    /// Return an iterator which will yield an infinite number of randomly
    /// generated items.
    ///
    /// # Example
    ///
    /// ```
    /// use std::rand::{task_rng, Rng};
    ///
    /// let mut rng = task_rng();
    /// let x = rng.gen_iter::<uint>().take(10).collect::<Vec<uint>>();
    /// println!("{}", x);
    /// println!("{}", rng.gen_iter::<(f64, bool)>().take(5)
    ///                   .collect::<Vec<(f64, bool)>>());
    /// ```
    fn gen_iter<'a, T: Rand>(&'a mut self) -> Generator<'a, T, Self> {
        Generator { rng: self }
    }

    /// Generate a random value in the range [`low`, `high`). Fails if
    /// `low >= high`.
    ///
    /// This is a convenience wrapper around
    /// `distributions::Range`. If this function will be called
    /// repeatedly with the same arguments, one should use `Range`, as
    /// that will amortize the computations that allow for perfect
    /// uniformity, as they only happen on initialization.
    ///
    /// # Example
    ///
    /// ```rust
    /// use std::rand::{task_rng, Rng};
    ///
    /// let mut rng = task_rng();
    /// let n: uint = rng.gen_range(0u, 10);
    /// println!("{}", n);
    /// let m: f64 = rng.gen_range(-40.0f64, 1.3e5f64);
    /// println!("{}", m);
    /// ```
    fn gen_range<T: PartialOrd + SampleRange>(&mut self, low: T, high: T) -> T {
        assert!(low < high, "Rng.gen_range called with low >= high");
        Range::new(low, high).ind_sample(self)
    }

    /// Return a bool with a 1 in n chance of true
    ///
    /// # Example
    ///
    /// ```rust
    /// use std::rand::{task_rng, Rng};
    ///
    /// let mut rng = task_rng();
    /// println!("{:b}", rng.gen_weighted_bool(3));
    /// ```
    fn gen_weighted_bool(&mut self, n: uint) -> bool {
        n == 0 || self.gen_range(0, n) == 0
    }

    /// Return an iterator of random characters from the set A-Z,a-z,0-9.
    ///
    /// # Example
    ///
    /// ```rust
    /// use std::rand::{task_rng, Rng};
    ///
    /// let s: String = task_rng().gen_ascii_chars().take(10).collect();
    /// println!("{}", s);
    /// ```
    fn gen_ascii_chars<'a>(&'a mut self) -> AsciiGenerator<'a, Self> {
        AsciiGenerator { rng: self }
    }

    /// Return a random element from `values`.
    ///
    /// Return `None` if `values` is empty.
    ///
    /// # Example
    ///
    /// ```
    /// use std::rand::{task_rng, Rng};
    ///
    /// let choices = [1i, 2, 4, 8, 16, 32];
    /// let mut rng = task_rng();
    /// println!("{}", rng.choose(choices));
    /// assert_eq!(rng.choose(choices[..0]), None);
    /// ```
    fn choose<'a, T>(&mut self, values: &'a [T]) -> Option<&'a T> {
        if values.is_empty() {
            None
        } else {
            Some(&values[self.gen_range(0u, values.len())])
        }
    }

    /// Deprecated name for `choose()`.
    #[deprecated = "replaced by .choose()"]
    fn choose_option<'a, T>(&mut self, values: &'a [T]) -> Option<&'a T> {
        self.choose(values)
    }

    /// Shuffle a mutable slice in place.
    ///
    /// # Example
    ///
    /// ```rust
    /// use std::rand::{task_rng, Rng};
    ///
    /// let mut rng = task_rng();
    /// let mut y = [1i, 2, 3];
    /// rng.shuffle(y);
    /// println!("{}", y.as_slice());
    /// rng.shuffle(y);
    /// println!("{}", y.as_slice());
    /// ```
    fn shuffle<T>(&mut self, values: &mut [T]) {
        let mut i = values.len();
        while i >= 2u {
            // invariant: elements with index >= i have been locked in place.
            i -= 1u;
            // lock element i in place.
            values.swap(i, self.gen_range(0u, i + 1u));
        }
    }
}

/// Iterator which will generate a stream of random items.
///
/// This iterator is created via the `gen_iter` method on `Rng`.
pub struct Generator<'a, T, R:'a> {
    rng: &'a mut R,
}

impl<'a, T: Rand, R: Rng> Iterator<T> for Generator<'a, T, R> {
    fn next(&mut self) -> Option<T> {
        Some(self.rng.gen())
    }
}

/// Iterator which will continuously generate random ascii characters.
///
/// This iterator is created via the `gen_ascii_chars` method on `Rng`.
pub struct AsciiGenerator<'a, R:'a> {
    rng: &'a mut R,
}

impl<'a, R: Rng> Iterator<char> for AsciiGenerator<'a, R> {
    fn next(&mut self) -> Option<char> {
        static GEN_ASCII_STR_CHARSET: &'static [u8] =
            b"ABCDEFGHIJKLMNOPQRSTUVWXYZ\
              abcdefghijklmnopqrstuvwxyz\
              0123456789";
        Some(*self.rng.choose(GEN_ASCII_STR_CHARSET).unwrap() as char)
    }
}

/// A random number generator that can be explicitly seeded to produce
/// the same stream of randomness multiple times.
pub trait SeedableRng<Seed>: Rng {
    /// Reseed an RNG with the given seed.
    ///
    /// # Example
    ///
    /// ```rust
    /// use std::rand::{Rng, SeedableRng, StdRng};
    ///
    /// let seed: &[_] = &[1, 2, 3, 4];
    /// let mut rng: StdRng = SeedableRng::from_seed(seed);
    /// println!("{}", rng.gen::<f64>());
    /// rng.reseed([5, 6, 7, 8]);
    /// println!("{}", rng.gen::<f64>());
    /// ```
    fn reseed(&mut self, Seed);

    /// Create a new RNG with the given seed.
    ///
    /// # Example
    ///
    /// ```rust
    /// use std::rand::{Rng, SeedableRng, StdRng};
    ///
    /// let seed: &[_] = &[1, 2, 3, 4];
    /// let mut rng: StdRng = SeedableRng::from_seed(seed);
    /// println!("{}", rng.gen::<f64>());
    /// ```
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

impl SeedableRng<[u32, .. 4]> for XorShiftRng {
    /// Reseed an XorShiftRng. This will fail if `seed` is entirely 0.
    fn reseed(&mut self, seed: [u32, .. 4]) {
        assert!(!seed.iter().all(|&x| x == 0),
                "XorShiftRng.reseed called with an all zero seed.");

        self.x = seed[0];
        self.y = seed[1];
        self.z = seed[2];
        self.w = seed[3];
    }

    /// Create a new XorShiftRng. This will fail if `seed` is entirely 0.
    fn from_seed(seed: [u32, .. 4]) -> XorShiftRng {
        assert!(!seed.iter().all(|&x| x == 0),
                "XorShiftRng::from_seed called with an all zero seed.");

        XorShiftRng {
            x: seed[0],
            y: seed[1],
            z: seed[2],
            w: seed[3]
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
        XorShiftRng { x: x, y: y, z: z, w: w }
    }
}

/// A wrapper for generating floating point numbers uniformly in the
/// open interval `(0,1)` (not including either endpoint).
///
/// Use `Closed01` for the closed interval `[0,1]`, and the default
/// `Rand` implementation for `f32` and `f64` for the half-open
/// `[0,1)`.
///
/// # Example
/// ```rust
/// use std::rand::{random, Open01};
///
/// let Open01(val) = random::<Open01<f32>>();
/// println!("f32 from (0,1): {}", val);
/// ```
pub struct Open01<F>(pub F);

/// A wrapper for generating floating point numbers uniformly in the
/// closed interval `[0,1]` (including both endpoints).
///
/// Use `Open01` for the closed interval `(0,1)`, and the default
/// `Rand` implementation of `f32` and `f64` for the half-open
/// `[0,1)`.
///
/// # Example
///
/// ```rust
/// use std::rand::{random, Closed01};
///
/// let Closed01(val) = random::<Closed01<f32>>();
/// println!("f32 from [0,1]: {}", val);
/// ```
pub struct Closed01<F>(pub F);

#[cfg(not(test))]
mod std {
    pub use core::{option, fmt}; // fail!()
}

#[cfg(test)]
mod test {
    use std::rand;

    pub struct MyRng<R> { inner: R }

    impl<R: rand::Rng> ::Rng for MyRng<R> {
        fn next_u32(&mut self) -> u32 {
            fn next<T: rand::Rng>(t: &mut T) -> u32 {
                use std::rand::Rng;
                t.next_u32()
            }
            next(&mut self.inner)
        }
    }

    pub fn rng() -> MyRng<rand::TaskRng> {
        MyRng { inner: rand::task_rng() }
    }

    pub fn weak_rng() -> MyRng<rand::XorShiftRng> {
        MyRng { inner: rand::weak_rng() }
    }
}
