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
//! # Task-local RNG
//!
//! There is built-in support for a RNG associated with each task stored
//! in task-local storage. This RNG can be accessed via `task_rng`, or
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
//!     Linux 3,17 added `getrandom(2)` system call which solves the issue: it blocks if entropy
//!     pool is not initialized yet, but it does not block once initialized.
//!     `OsRng` tries to use `getrandom(2)` if available, and use `/dev/urandom` fallback if not.
//!     If an application does not have `getrandom` and likely to be run soon after first booting,
//!     or on a system with very few entropy sources, one should consider using `/dev/random` via
//!     `ReaderRng`.
//! -   On some systems (e.g. FreeBSD, OpenBSD and Mac OS X) there is no difference
//!     between the two sources. (Also note that, on some systems e.g. FreeBSD, both `/dev/random`
//!     and `/dev/urandom` may block once if the CSPRNG has not seeded yet.)
//!
//! # Examples
//!
//! ```rust
//! use std::rand;
//! use std::rand::Rng;
//!
//! let mut rng = rand::task_rng();
//! if rng.gen() { // random bool
//!     println!("int: {}, uint: {}", rng.gen::<int>(), rng.gen::<uint>())
//! }
//! ```
//!
//! ```rust
//! use std::rand;
//!
//! let tuple = rand::random::<(f64, char)>();
//! println!("{}", tuple)
//! ```
//!
//! ## Monte Carlo estimation of π
//!
//! For this example, imagine we have a square with sides of length 2 and a unit
//! circle, both centered at the origin. Since the area of a unit circle is π,
//! we have:
//!
//! ```notrust
//!     (area of unit circle) / (area of square) = π / 4
//! ```
//!
//! So if we sample many points randomly from the square, roughly π / 4 of them
//! should be inside the circle.
//!
//! We can use the above fact to estimate the value of π: pick many points in the
//! square at random, calculate the fraction that fall within the circle, and
//! multiply this fraction by 4.
//!
//! ```
//! use std::rand;
//! use std::rand::distributions::{IndependentSample, Range};
//!
//! fn main() {
//!    let between = Range::new(-1f64, 1.);
//!    let mut rng = rand::task_rng();
//!
//!    let total = 1_000_000u;
//!    let mut in_circle = 0u;
//!
//!    for _ in range(0u, total) {
//!        let a = between.ind_sample(&mut rng);
//!        let b = between.ind_sample(&mut rng);
//!        if a*a + b*b <= 1. {
//!            in_circle += 1;
//!        }
//!    }
//!
//!    // prints something close to 3.14159...
//!    println!("{}", 4. * (in_circle as f64) / (total as f64));
//! }
//! ```
//!
//! ## Monty Hall Problem
//!
//! This is a simulation of the [Monty Hall Problem][]:
//!
//! > Suppose you're on a game show, and you're given the choice of three doors:
//! > Behind one door is a car; behind the others, goats. You pick a door, say No. 1,
//! > and the host, who knows what's behind the doors, opens another door, say No. 3,
//! > which has a goat. He then says to you, "Do you want to pick door No. 2?"
//! > Is it to your advantage to switch your choice?
//!
//! The rather unintuitive answer is that you will have a 2/3 chance of winning if
//! you switch and a 1/3 chance of winning of you don't, so it's better to switch.
//!
//! This program will simulate the game show and with large enough simulation steps
//! it will indeed confirm that it is better to switch.
//!
//! [Monty Hall Problem]: http://en.wikipedia.org/wiki/Monty_Hall_problem
//!
//! ```
//! use std::rand;
//! use std::rand::Rng;
//! use std::rand::distributions::{IndependentSample, Range};
//!
//! struct SimulationResult {
//!     win: bool,
//!     switch: bool,
//! }
//!
//! // Run a single simulation of the Monty Hall problem.
//! fn simulate<R: Rng>(random_door: &Range<uint>, rng: &mut R) -> SimulationResult {
//!     let car = random_door.ind_sample(rng);
//!
//!     // This is our initial choice
//!     let mut choice = random_door.ind_sample(rng);
//!
//!     // The game host opens a door
//!     let open = game_host_open(car, choice, rng);
//!
//!     // Shall we switch?
//!     let switch = rng.gen();
//!     if switch {
//!         choice = switch_door(choice, open);
//!     }
//!
//!     SimulationResult { win: choice == car, switch: switch }
//! }
//!
//! // Returns the door the game host opens given our choice and knowledge of
//! // where the car is. The game host will never open the door with the car.
//! fn game_host_open<R: Rng>(car: uint, choice: uint, rng: &mut R) -> uint {
//!     let choices = free_doors(&[car, choice]);
//!     rand::sample(rng, choices.into_iter(), 1)[0]
//! }
//!
//! // Returns the door we switch to, given our current choice and
//! // the open door. There will only be one valid door.
//! fn switch_door(choice: uint, open: uint) -> uint {
//!     free_doors(&[choice, open])[0]
//! }
//!
//! fn free_doors(blocked: &[uint]) -> Vec<uint> {
//!     range(0u, 3).filter(|x| !blocked.contains(x)).collect()
//! }
//!
//! fn main() {
//!     // The estimation will be more accurate with more simulations
//!     let num_simulations = 10000u;
//!
//!     let mut rng = rand::task_rng();
//!     let random_door = Range::new(0u, 3);
//!
//!     let (mut switch_wins, mut switch_losses) = (0u, 0u);
//!     let (mut keep_wins, mut keep_losses) = (0u, 0u);
//!
//!     println!("Running {} simulations...", num_simulations);
//!     for _ in range(0, num_simulations) {
//!         let result = simulate(&random_door, &mut rng);
//!
//!         match (result.win, result.switch) {
//!             (true, true) => switch_wins += 1,
//!             (true, false) => keep_wins += 1,
//!             (false, true) => switch_losses += 1,
//!             (false, false) => keep_losses += 1,
//!         }
//!     }
//!
//!     let total_switches = switch_wins + switch_losses;
//!     let total_keeps = keep_wins + keep_losses;
//!
//!     println!("Switched door {} times with {} wins and {} losses",
//!              total_switches, switch_wins, switch_losses);
//!
//!     println!("Kept our choice {} times with {} wins and {} losses",
//!              total_keeps, keep_wins, keep_losses);
//!
//!     // With a large number of simulations, the values should converge to
//!     // 0.667 and 0.333 respectively.
//!     println!("Estimated chance to win if we switch: {}",
//!              switch_wins as f32 / total_switches as f32);
//!     println!("Estimated chance to win if we don't: {}",
//!              keep_wins as f32 / total_keeps as f32);
//! }
//! ```

#![experimental]

use cell::RefCell;
use clone::Clone;
use io::IoResult;
use iter::Iterator;
use mem;
use rc::Rc;
use result::{Ok, Err};
use vec::Vec;

#[cfg(not(target_word_size="64"))]
use core_rand::IsaacRng as IsaacWordRng;
#[cfg(target_word_size="64")]
use core_rand::Isaac64Rng as IsaacWordRng;

pub use core_rand::{Rand, Rng, SeedableRng, Open01, Closed01};
pub use core_rand::{XorShiftRng, IsaacRng, Isaac64Rng, ChaChaRng};
pub use core_rand::{distributions, reseeding};
pub use rand::os::OsRng;

pub mod os;
pub mod reader;

/// The standard RNG. This is designed to be efficient on the current
/// platform.
pub struct StdRng { rng: IsaacWordRng }

impl StdRng {
    /// Create a randomly seeded instance of `StdRng`.
    ///
    /// This is a very expensive operation as it has to read
    /// randomness from the operating system and use this in an
    /// expensive seeding operation. If one is only generating a small
    /// number of random numbers, or doesn't need the utmost speed for
    /// generating each number, `task_rng` and/or `random` may be more
    /// appropriate.
    ///
    /// Reading the randomness from the OS may fail, and any error is
    /// propagated via the `IoResult` return value.
    pub fn new() -> IoResult<StdRng> { unimplemented!() }
}

impl Rng for StdRng {
    #[inline]
    fn next_u32(&mut self) -> u32 { unimplemented!() }

    #[inline]
    fn next_u64(&mut self) -> u64 { unimplemented!() }
}

impl<'a> SeedableRng<&'a [uint]> for StdRng {
    fn reseed(&mut self, seed: &'a [uint]) { unimplemented!() }

    fn from_seed(seed: &'a [uint]) -> StdRng { unimplemented!() }
}

/// Create a weak random number generator with a default algorithm and seed.
///
/// It returns the fastest `Rng` algorithm currently available in Rust without
/// consideration for cryptography or security. If you require a specifically
/// seeded `Rng` for consistency over time you should pick one algorithm and
/// create the `Rng` yourself.
///
/// This will read randomness from the operating system to seed the
/// generator.
pub fn weak_rng() -> XorShiftRng { unimplemented!() }

/// Controls how the task-local RNG is reseeded.
struct TaskRngReseeder;

impl reseeding::Reseeder<StdRng> for TaskRngReseeder {
    fn reseed(&mut self, rng: &mut StdRng) { unimplemented!() }
}
static TASK_RNG_RESEED_THRESHOLD: uint = 32_768;
type TaskRngInner = reseeding::ReseedingRng<StdRng, TaskRngReseeder>;

/// The task-local RNG.
pub struct TaskRng {
    rng: Rc<RefCell<TaskRngInner>>,
}

/// Retrieve the lazily-initialized task-local random number
/// generator, seeded by the system. Intended to be used in method
/// chaining style, e.g. `task_rng().gen::<int>()`.
///
/// The RNG provided will reseed itself from the operating system
/// after generating a certain amount of randomness.
///
/// The internal RNG used is platform and architecture dependent, even
/// if the operating system random number generator is rigged to give
/// the same sequence always. If absolute consistency is required,
/// explicitly select an RNG, e.g. `IsaacRng` or `Isaac64Rng`.
pub fn task_rng() -> TaskRng { unimplemented!() }

impl Rng for TaskRng {
    fn next_u32(&mut self) -> u32 { unimplemented!() }

    fn next_u64(&mut self) -> u64 { unimplemented!() }

    #[inline]
    fn fill_bytes(&mut self, bytes: &mut [u8]) { unimplemented!() }
}

/// Generates a random value using the task-local random number generator.
///
/// `random()` can generate various types of random things, and so may require
/// type hinting to generate the specific type you want.
///
/// # Examples
///
/// ```rust
/// use std::rand;
///
/// let x = rand::random();
/// println!("{}", 2u * x);
///
/// let y = rand::random::<f64>();
/// println!("{}", y);
///
/// if rand::random() { // generates a boolean
///     println!("Better lucky than good!");
/// }
/// ```
#[inline]
pub fn random<T: Rand>() -> T { unimplemented!() }

/// Randomly sample up to `amount` elements from an iterator.
///
/// # Example
///
/// ```rust
/// use std::rand::{task_rng, sample};
///
/// let mut rng = task_rng();
/// let sample = sample(&mut rng, range(1i, 100), 5);
/// println!("{}", sample);
/// ```
pub fn sample<T, I: Iterator<T>, R: Rng>(rng: &mut R,
                                         mut iter: I,
                                         amount: uint) -> Vec<T> { unimplemented!() }
