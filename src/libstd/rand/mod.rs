// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/*!

Utilities for random number generation

The key functions are `random()` and `Rng::gen()`. These are polymorphic
and so can be used to generate any type that implements `Rand`. Type inference
means that often a simple call to `rand::random()` or `rng.gen()` will
suffice, but sometimes an annotation is required, e.g. `rand::random::<f64>()`.

See the `distributions` submodule for sampling random numbers from
distributions like normal and exponential.

# Task-local RNG

There is built-in support for a RNG associated with each task stored
in task-local storage. This RNG can be accessed via `task_rng`, or
used implicitly via `random`. This RNG is normally randomly seeded
from an operating-system source of randomness, e.g. `/dev/urandom` on
Unix systems, and will automatically reseed itself from this source
after generating 32 KiB of random data.

# Cryptographic security

An application that requires an entropy source for cryptographic purposes
must use `OsRng`, which reads randomness from the source that the operating
system provides (e.g. `/dev/urandom` on Unixes or `CryptGenRandom()` on Windows).
The other random number generators provided by this module are not suitable
for such purposes.

*Note*: many Unix systems provide `/dev/random` as well as `/dev/urandom`.
This module uses `/dev/urandom` for the following reasons:

-   On Linux, `/dev/random` may block if entropy pool is empty; `/dev/urandom` will not block.
    This does not mean that `/dev/random` provides better output than
    `/dev/urandom`; the kernel internally runs a cryptographically secure pseudorandom
    number generator (CSPRNG) based on entropy pool for random number generation,
    so the "quality" of `/dev/random` is not better than `/dev/urandom` in most cases.
    However, this means that `/dev/urandom` can yield somewhat predictable randomness
    if the entropy pool is very small, such as immediately after first booting.
    If an application likely to be run soon after first booting, or on a system with very
    few entropy sources, one should consider using `/dev/random` via `ReaderRng`.
-   On some systems (e.g. FreeBSD, OpenBSD and Mac OS X) there is no difference
    between the two sources. (Also note that, on some systems e.g. FreeBSD, both `/dev/random`
    and `/dev/urandom` may block once if the CSPRNG has not seeded yet.)

# Examples

```rust
use std::rand;
use std::rand::Rng;

let mut rng = rand::task_rng();
if rng.gen() { // bool
    println!("int: {}, uint: {}", rng.gen::<int>(), rng.gen::<uint>())
}
```

```rust
use std::rand;

let tuple = rand::random::<(f64, char)>();
println!("{}", tuple)
```
*/

use cell::RefCell;
use clone::Clone;
use io::IoResult;
use iter::Iterator;
use mem;
use option::{Some, None};
use rc::Rc;
use result::{Ok, Err};
use vec::Vec;

#[cfg(not(target_word_size="64"))]
use IsaacWordRng = core_rand::IsaacRng;
#[cfg(target_word_size="64")]
use IsaacWordRng = core_rand::Isaac64Rng;

pub use core_rand::{Rand, Rng, SeedableRng, Open01, Closed01};
pub use core_rand::{XorShiftRng, IsaacRng, Isaac64Rng};
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
    pub fn new() -> IoResult<StdRng> {
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

impl<'a> SeedableRng<&'a [uint]> for StdRng {
    fn reseed(&mut self, seed: &'a [uint]) {
        // the internal RNG can just be seeded from the above
        // randomness.
        self.rng.reseed(unsafe {mem::transmute(seed)})
    }

    fn from_seed(seed: &'a [uint]) -> StdRng {
        StdRng { rng: SeedableRng::from_seed(unsafe {mem::transmute(seed)}) }
    }
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
pub fn weak_rng() -> XorShiftRng {
    match OsRng::new() {
        Ok(mut r) => r.gen(),
        Err(e) => fail!("weak_rng: failed to create seeded RNG: {}", e)
    }
}

/// Controls how the task-local RNG is reseeded.
struct TaskRngReseeder;

impl reseeding::Reseeder<StdRng> for TaskRngReseeder {
    fn reseed(&mut self, rng: &mut StdRng) {
        *rng = match StdRng::new() {
            Ok(r) => r,
            Err(e) => fail!("could not reseed task_rng: {}", e)
        }
    }
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
pub fn task_rng() -> TaskRng {
    // used to make space in TLS for a random number generator
    local_data_key!(TASK_RNG_KEY: Rc<RefCell<TaskRngInner>>)

    match TASK_RNG_KEY.get() {
        None => {
            let r = match StdRng::new() {
                Ok(r) => r,
                Err(e) => fail!("could not initialize task_rng: {}", e)
            };
            let rng = reseeding::ReseedingRng::new(r,
                                                   TASK_RNG_RESEED_THRESHOLD,
                                                   TaskRngReseeder);
            let rng = Rc::new(RefCell::new(rng));
            TASK_RNG_KEY.replace(Some(rng.clone()));

            TaskRng { rng: rng }
        }
        Some(rng) => TaskRng { rng: rng.clone() }
    }
}

impl Rng for TaskRng {
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

/// Generate a random value using the task-local random number
/// generator.
///
/// # Example
///
/// ```rust
/// use std::rand::random;
///
/// if random() {
///     let x = random();
///     println!("{}", 2u * x);
/// } else {
///     println!("{}", random::<f64>());
/// }
/// ```
#[inline]
pub fn random<T: Rand>() -> T {
    task_rng().gen()
}

/// Randomly sample up to `n` elements from an iterator.
///
/// # Example
///
/// ```rust
/// use std::rand::{task_rng, sample};
///
/// let mut rng = task_rng();
/// let sample = sample(&mut rng, range(1, 100), 5);
/// println!("{}", sample);
/// ```
pub fn sample<T, I: Iterator<T>, R: Rng>(rng: &mut R,
                                         mut iter: I,
                                         amt: uint) -> Vec<T> {
    let mut reservoir: Vec<T> = iter.by_ref().take(amt).collect();
    for (i, elem) in iter.enumerate() {
        let k = rng.gen_range(0, i + 1 + amt);
        if k < amt {
            *reservoir.get_mut(k) = elem;
        }
    }
    return reservoir;
}

#[cfg(test)]
mod test {
    use prelude::*;
    use super::{Rng, task_rng, random, SeedableRng, StdRng, sample};
    use iter::order;

    struct ConstRng { i: u64 }
    impl Rng for ConstRng {
        fn next_u32(&mut self) -> u32 { self.i as u32 }
        fn next_u64(&mut self) -> u64 { self.i }

        // no fill_bytes on purpose
    }

    #[test]
    fn test_fill_bytes_default() {
        let mut r = ConstRng { i: 0x11_22_33_44_55_66_77_88 };

        // check every remainder mod 8, both in small and big vectors.
        let lengths = [0, 1, 2, 3, 4, 5, 6, 7,
                       80, 81, 82, 83, 84, 85, 86, 87];
        for &n in lengths.iter() {
            let mut v = Vec::from_elem(n, 0u8);
            r.fill_bytes(v.as_mut_slice());

            // use this to get nicer error messages.
            for (i, &byte) in v.iter().enumerate() {
                if byte == 0 {
                    fail!("byte {} of {} is zero", i, n)
                }
            }
        }
    }

    #[test]
    fn test_gen_range() {
        let mut r = task_rng();
        for _ in range(0, 1000) {
            let a = r.gen_range(-3i, 42);
            assert!(a >= -3 && a < 42);
            assert_eq!(r.gen_range(0, 1), 0);
            assert_eq!(r.gen_range(-12, -11), -12);
        }

        for _ in range(0, 1000) {
            let a = r.gen_range(10, 42);
            assert!(a >= 10 && a < 42);
            assert_eq!(r.gen_range(0, 1), 0);
            assert_eq!(r.gen_range(3_000_000u, 3_000_001), 3_000_000);
        }

    }

    #[test]
    #[should_fail]
    fn test_gen_range_fail_int() {
        let mut r = task_rng();
        r.gen_range(5i, -2);
    }

    #[test]
    #[should_fail]
    fn test_gen_range_fail_uint() {
        let mut r = task_rng();
        r.gen_range(5u, 2u);
    }

    #[test]
    fn test_gen_f64() {
        let mut r = task_rng();
        let a = r.gen::<f64>();
        let b = r.gen::<f64>();
        debug!("{}", (a, b));
    }

    #[test]
    fn test_gen_weighted_bool() {
        let mut r = task_rng();
        assert_eq!(r.gen_weighted_bool(0u), true);
        assert_eq!(r.gen_weighted_bool(1u), true);
    }

    #[test]
    fn test_gen_ascii_str() {
        let mut r = task_rng();
        assert_eq!(r.gen_ascii_chars().take(0).count(), 0u);
        assert_eq!(r.gen_ascii_chars().take(10).count(), 10u);
        assert_eq!(r.gen_ascii_chars().take(16).count(), 16u);
    }

    #[test]
    fn test_gen_vec() {
        let mut r = task_rng();
        assert_eq!(r.gen_iter::<u8>().take(0).count(), 0u);
        assert_eq!(r.gen_iter::<u8>().take(10).count(), 10u);
        assert_eq!(r.gen_iter::<f64>().take(16).count(), 16u);
    }

    #[test]
    fn test_choose() {
        let mut r = task_rng();
        assert_eq!(r.choose([1, 1, 1]).map(|&x|x), Some(1));

        let v: &[int] = &[];
        assert_eq!(r.choose(v), None);
    }

    #[test]
    fn test_shuffle() {
        let mut r = task_rng();
        let empty: &mut [int] = &mut [];
        r.shuffle(empty);
        let mut one = [1];
        r.shuffle(one);
        assert_eq!(one.as_slice(), &[1]);

        let mut two = [1, 2];
        r.shuffle(two);
        assert!(two == [1, 2] || two == [2, 1]);

        let mut x = [1, 1, 1];
        r.shuffle(x);
        assert_eq!(x.as_slice(), &[1, 1, 1]);
    }

    #[test]
    fn test_task_rng() {
        let mut r = task_rng();
        r.gen::<int>();
        let mut v = [1, 1, 1];
        r.shuffle(v);
        assert_eq!(v.as_slice(), &[1, 1, 1]);
        assert_eq!(r.gen_range(0u, 1u), 0u);
    }

    #[test]
    fn test_random() {
        // not sure how to test this aside from just getting some values
        let _n : uint = random();
        let _f : f32 = random();
        let _o : Option<Option<i8>> = random();
        let _many : ((),
                     (uint,
                      int,
                      Option<(u32, (bool,))>),
                     (u8, i8, u16, i16, u32, i32, u64, i64),
                     (f32, (f64, (f64,)))) = random();
    }

    #[test]
    fn test_sample() {
        let min_val = 1;
        let max_val = 100;

        let mut r = task_rng();
        let vals = range(min_val, max_val).collect::<Vec<int>>();
        let small_sample = sample(&mut r, vals.iter(), 5);
        let large_sample = sample(&mut r, vals.iter(), vals.len() + 5);

        assert_eq!(small_sample.len(), 5);
        assert_eq!(large_sample.len(), vals.len());

        assert!(small_sample.iter().all(|e| {
            **e >= min_val && **e <= max_val
        }));
    }

    #[test]
    fn test_std_rng_seeded() {
        let s = task_rng().gen_iter::<uint>().take(256).collect::<Vec<uint>>();
        let mut ra: StdRng = SeedableRng::from_seed(s.as_slice());
        let mut rb: StdRng = SeedableRng::from_seed(s.as_slice());
        assert!(order::equals(ra.gen_ascii_chars().take(100),
                              rb.gen_ascii_chars().take(100)));
    }

    #[test]
    fn test_std_rng_reseed() {
        let s = task_rng().gen_iter::<uint>().take(256).collect::<Vec<uint>>();
        let mut r: StdRng = SeedableRng::from_seed(s.as_slice());
        let string1 = r.gen_ascii_chars().take(100).collect::<String>();

        r.reseed(s.as_slice());

        let string2 = r.gen_ascii_chars().take(100).collect::<String>();
        assert_eq!(string1, string2);
    }
}

#[cfg(test)]
static RAND_BENCH_N: u64 = 100;

#[cfg(test)]
mod bench {
    extern crate test;
    use prelude::*;

    use self::test::Bencher;
    use super::{XorShiftRng, StdRng, IsaacRng, Isaac64Rng, Rng, RAND_BENCH_N};
    use super::{OsRng, weak_rng};
    use mem::size_of;

    #[bench]
    fn rand_xorshift(b: &mut Bencher) {
        let mut rng: XorShiftRng = OsRng::new().unwrap().gen();
        b.iter(|| {
            for _ in range(0, RAND_BENCH_N) {
                rng.gen::<uint>();
            }
        });
        b.bytes = size_of::<uint>() as u64 * RAND_BENCH_N;
    }

    #[bench]
    fn rand_isaac(b: &mut Bencher) {
        let mut rng: IsaacRng = OsRng::new().unwrap().gen();
        b.iter(|| {
            for _ in range(0, RAND_BENCH_N) {
                rng.gen::<uint>();
            }
        });
        b.bytes = size_of::<uint>() as u64 * RAND_BENCH_N;
    }

    #[bench]
    fn rand_isaac64(b: &mut Bencher) {
        let mut rng: Isaac64Rng = OsRng::new().unwrap().gen();
        b.iter(|| {
            for _ in range(0, RAND_BENCH_N) {
                rng.gen::<uint>();
            }
        });
        b.bytes = size_of::<uint>() as u64 * RAND_BENCH_N;
    }

    #[bench]
    fn rand_std(b: &mut Bencher) {
        let mut rng = StdRng::new().unwrap();
        b.iter(|| {
            for _ in range(0, RAND_BENCH_N) {
                rng.gen::<uint>();
            }
        });
        b.bytes = size_of::<uint>() as u64 * RAND_BENCH_N;
    }

    #[bench]
    fn rand_shuffle_100(b: &mut Bencher) {
        let mut rng = weak_rng();
        let x : &mut[uint] = [1,..100];
        b.iter(|| {
            rng.shuffle(x);
        })
    }
}
