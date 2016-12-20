// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Utilities for random number generation
//!
//! The key functions are `random()` and `Rng::gen()`. These are polymorphic
//! and so can be used to generate any type that implements `Rand`. Type inference
//! means that often a simple call to `rand::random()` or `rng.gen()` will
//! suffice, but sometimes an annotation is required, e.g. `rand::random::<f64>()`.
//!
//! See the `distributions` submodule for sampling random numbers from
//! distributions like normal and exponential.
//!
//! # Thread-local RNG
//!
//! There is built-in support for a RNG associated with each thread stored
//! in thread-local storage. This RNG can be accessed via `thread_rng`, or
//! used implicitly via `random`. This RNG is normally randomly seeded
//! from an operating-system source of randomness, e.g. `/dev/urandom` on
//! Unix systems, and will automatically reseed itself from this source
//! after generating 32 KiB of random data.
//!
//! # Cryptographic security
//!
//! An application that requires an entropy source for cryptographic purposes
//! must use `OsRng`, which reads randomness from the source that the operating
//! system provides (e.g. `/dev/urandom` on Unixes or `CryptGenRandom()` on Windows).
//! The other random number generators provided by this module are not suitable
//! for such purposes.
//!
//! *Note*: many Unix systems provide `/dev/random` as well as `/dev/urandom`.
//! This module uses `/dev/urandom` for the following reasons:
//!
//! -   On Linux, `/dev/random` may block if entropy pool is empty; `/dev/urandom` will not block.
//!     This does not mean that `/dev/random` provides better output than
//!     `/dev/urandom`; the kernel internally runs a cryptographically secure pseudorandom
//!     number generator (CSPRNG) based on entropy pool for random number generation,
//!     so the "quality" of `/dev/random` is not better than `/dev/urandom` in most cases.
//!     However, this means that `/dev/urandom` can yield somewhat predictable randomness
//!     if the entropy pool is very small, such as immediately after first booting.
//!     Linux 3.17 added the `getrandom(2)` system call which solves the issue: it blocks if entropy
//!     pool is not initialized yet, but it does not block once initialized.
//!     `getrandom(2)` was based on `getentropy(2)`, an existing system call in OpenBSD.
//!     `OsRng` tries to use `getrandom(2)` if available, and use `/dev/urandom` fallback if not.
//!     If an application does not have `getrandom` and likely to be run soon after first booting,
//!     or on a system with very few entropy sources, one should consider using `/dev/random` via
//!     `ReaderRng`.
//! -   On some systems (e.g. FreeBSD, OpenBSD and Mac OS X) there is no difference
//!     between the two sources. (Also note that, on some systems e.g. FreeBSD, both `/dev/random`
//!     and `/dev/urandom` may block once if the CSPRNG has not seeded yet.)

#![unstable(feature = "rand", issue = "0")]

use cell::RefCell;
use fmt;
use io;
use mem;
use rc::Rc;
use sys;

#[cfg(target_pointer_width = "32")]
use core_rand::IsaacRng as IsaacWordRng;
#[cfg(target_pointer_width = "64")]
use core_rand::Isaac64Rng as IsaacWordRng;

pub use core_rand::{Rand, Rng, SeedableRng};
pub use core_rand::{XorShiftRng, IsaacRng, Isaac64Rng};
pub use core_rand::reseeding;

pub mod reader;

/// The standard RNG. This is designed to be efficient on the current
/// platform.
#[derive(Copy, Clone)]
pub struct StdRng {
    rng: IsaacWordRng,
}

impl StdRng {
    /// Create a randomly seeded instance of `StdRng`.
    ///
    /// This is a very expensive operation as it has to read
    /// randomness from the operating system and use this in an
    /// expensive seeding operation. If one is only generating a small
    /// number of random numbers, or doesn't need the utmost speed for
    /// generating each number, `thread_rng` and/or `random` may be more
    /// appropriate.
    ///
    /// Reading the randomness from the OS may fail, and any error is
    /// propagated via the `io::Result` return value.
    pub fn new() -> io::Result<StdRng> {
        OsRng::new().map(|mut r| StdRng { rng: r.gen() })
    }
}

impl Rng for StdRng {
    #[inline]
    fn next_u32(&mut self) -> u32 {
        self.rng.next_u32()
    }

    #[inline]
    fn next_u64(&mut self) -> u64 {
        self.rng.next_u64()
    }
}

impl<'a> SeedableRng<&'a [usize]> for StdRng {
    fn reseed(&mut self, seed: &'a [usize]) {
        // the internal RNG can just be seeded from the above
        // randomness.
        self.rng.reseed(unsafe {mem::transmute(seed)})
    }

    fn from_seed(seed: &'a [usize]) -> StdRng {
        StdRng { rng: SeedableRng::from_seed(unsafe {mem::transmute(seed)}) }
    }
}

/// Controls how the thread-local RNG is reseeded.
struct ThreadRngReseeder;

impl reseeding::Reseeder<StdRng> for ThreadRngReseeder {
    fn reseed(&mut self, rng: &mut StdRng) {
        *rng = match StdRng::new() {
            Ok(r) => r,
            Err(e) => panic!("could not reseed thread_rng: {}", e)
        }
    }
}
const THREAD_RNG_RESEED_THRESHOLD: usize = 32_768;
type ThreadRngInner = reseeding::ReseedingRng<StdRng, ThreadRngReseeder>;

/// The thread-local RNG.
#[derive(Clone)]
pub struct ThreadRng {
    rng: Rc<RefCell<ThreadRngInner>>,
}

impl fmt::Debug for ThreadRng {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.pad("ThreadRng { .. }")
    }
}

/// Retrieve the lazily-initialized thread-local random number
/// generator, seeded by the system. Intended to be used in method
/// chaining style, e.g. `thread_rng().gen::<isize>()`.
///
/// The RNG provided will reseed itself from the operating system
/// after generating a certain amount of randomness.
///
/// The internal RNG used is platform and architecture dependent, even
/// if the operating system random number generator is rigged to give
/// the same sequence always. If absolute consistency is required,
/// explicitly select an RNG, e.g. `IsaacRng` or `Isaac64Rng`.
pub fn thread_rng() -> ThreadRng {
    // used to make space in TLS for a random number generator
    thread_local!(static THREAD_RNG_KEY: Rc<RefCell<ThreadRngInner>> = {
        let r = match StdRng::new() {
            Ok(r) => r,
            Err(e) => panic!("could not initialize thread_rng: {}", e)
        };
        let rng = reseeding::ReseedingRng::new(r,
                                               THREAD_RNG_RESEED_THRESHOLD,
                                               ThreadRngReseeder);
        Rc::new(RefCell::new(rng))
    });

    ThreadRng { rng: THREAD_RNG_KEY.with(|t| t.clone()) }
}

impl Rng for ThreadRng {
    fn next_u32(&mut self) -> u32 {
        self.rng.borrow_mut().next_u32()
    }

    fn next_u64(&mut self) -> u64 {
        self.rng.borrow_mut().next_u64()
    }

    #[inline]
    fn fill_bytes(&mut self, bytes: &mut [u8]) {
        self.rng.borrow_mut().fill_bytes(bytes)
    }
}

/// A random number generator that retrieves randomness straight from
/// the operating system. Platform sources:
///
/// - Unix-like systems (Linux, Android, Mac OSX): read directly from
///   `/dev/urandom`, or from `getrandom(2)` system call if available.
/// - Windows: calls `CryptGenRandom`, using the default cryptographic
///   service provider with the `PROV_RSA_FULL` type.
/// - iOS: calls SecRandomCopyBytes as /dev/(u)random is sandboxed.
/// - OpenBSD: uses the `getentropy(2)` system call.
///
/// This does not block.
pub struct OsRng(sys::rand::OsRng);

impl OsRng {
    /// Create a new `OsRng`.
    pub fn new() -> io::Result<OsRng> {
        sys::rand::OsRng::new().map(OsRng)
    }
}

impl Rng for OsRng {
    #[inline]
    fn next_u32(&mut self) -> u32 {
        self.0.next_u32()
    }

    #[inline]
    fn next_u64(&mut self) -> u64 {
        self.0.next_u64()
    }

    #[inline]
    fn fill_bytes(&mut self, bytes: &mut [u8]) {
        self.0.fill_bytes(bytes)
    }
}


#[cfg(test)]
mod tests {
    use sync::mpsc::channel;
    use rand::Rng;
    use super::OsRng;
    use thread;

    #[test]
    fn test_os_rng() {
        let mut r = OsRng::new().unwrap();

        r.next_u32();
        r.next_u64();

        let mut v = [0; 1000];
        r.fill_bytes(&mut v);
    }

    #[test]
    #[cfg_attr(target_os = "emscripten", ignore)]
    fn test_os_rng_tasks() {

        let mut txs = vec![];
        for _ in 0..20 {
            let (tx, rx) = channel();
            txs.push(tx);

            thread::spawn(move|| {
                // wait until all the threads are ready to go.
                rx.recv().unwrap();

                // deschedule to attempt to interleave things as much
                // as possible (XXX: is this a good test?)
                let mut r = OsRng::new().unwrap();
                thread::yield_now();
                let mut v = [0; 1000];

                for _ in 0..100 {
                    r.next_u32();
                    thread::yield_now();
                    r.next_u64();
                    thread::yield_now();
                    r.fill_bytes(&mut v);
                    thread::yield_now();
                }
            });
        }

        // start all the threads
        for tx in &txs {
            tx.send(()).unwrap();
        }
    }
}
