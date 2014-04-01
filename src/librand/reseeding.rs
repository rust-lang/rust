// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! A wrapper around another RNG that reseeds it after it
//! generates a certain number of random bytes.

use std::default::Default;
use {Rng, SeedableRng};

/// How many bytes of entropy the underling RNG is allowed to generate
/// before it is reseeded.
static DEFAULT_GENERATION_THRESHOLD: uint = 32 * 1024;

/// A wrapper around any RNG which reseeds the underlying RNG after it
/// has generated a certain number of random bytes.
pub struct ReseedingRng<R, Rsdr> {
    rng: R,
    generation_threshold: uint,
    bytes_generated: uint,
    /// Controls the behaviour when reseeding the RNG.
    pub reseeder: Rsdr,
}

impl<R: Rng, Rsdr: Reseeder<R>> ReseedingRng<R, Rsdr> {
    /// Create a new `ReseedingRng` with the given parameters.
    ///
    /// # Arguments
    ///
    /// * `rng`: the random number generator to use.
    /// * `generation_threshold`: the number of bytes of entropy at which to reseed the RNG.
    /// * `reseeder`: the reseeding object to use.
    pub fn new(rng: R, generation_threshold: uint, reseeder: Rsdr) -> ReseedingRng<R,Rsdr> {
        ReseedingRng {
            rng: rng,
            generation_threshold: generation_threshold,
            bytes_generated: 0,
            reseeder: reseeder
        }
    }

    /// Reseed the internal RNG if the number of bytes that have been
    /// generated exceed the threshold.
    pub fn reseed_if_necessary(&mut self) {
        if self.bytes_generated >= self.generation_threshold {
            self.reseeder.reseed(&mut self.rng);
            self.bytes_generated = 0;
        }
    }
}


impl<R: Rng, Rsdr: Reseeder<R>> Rng for ReseedingRng<R, Rsdr> {
    fn next_u32(&mut self) -> u32 {
        self.reseed_if_necessary();
        self.bytes_generated += 4;
        self.rng.next_u32()
    }

    fn next_u64(&mut self) -> u64 {
        self.reseed_if_necessary();
        self.bytes_generated += 8;
        self.rng.next_u64()
    }

    fn fill_bytes(&mut self, dest: &mut [u8]) {
        self.reseed_if_necessary();
        self.bytes_generated += dest.len();
        self.rng.fill_bytes(dest)
    }
}

impl<S, R: SeedableRng<S>, Rsdr: Reseeder<R>>
     SeedableRng<(Rsdr, S)> for ReseedingRng<R, Rsdr> {
    fn reseed(&mut self, (rsdr, seed): (Rsdr, S)) {
        self.rng.reseed(seed);
        self.reseeder = rsdr;
        self.bytes_generated = 0;
    }
    /// Create a new `ReseedingRng` from the given reseeder and
    /// seed. This uses a default value for `generation_threshold`.
    fn from_seed((rsdr, seed): (Rsdr, S)) -> ReseedingRng<R, Rsdr> {
        ReseedingRng {
            rng: SeedableRng::from_seed(seed),
            generation_threshold: DEFAULT_GENERATION_THRESHOLD,
            bytes_generated: 0,
            reseeder: rsdr
        }
    }
}

/// Something that can be used to reseed an RNG via `ReseedingRng`.
///
/// # Example
///
/// ```rust
/// use rand::{Rng, SeedableRng, StdRng};
/// use rand::reseeding::{Reseeder, ReseedingRng};
///
/// struct TickTockReseeder { tick: bool }
/// impl Reseeder<StdRng> for TickTockReseeder {
///     fn reseed(&mut self, rng: &mut StdRng) {
///         let val = if self.tick {0} else {1};
///         rng.reseed(&[val]);
///         self.tick = !self.tick;
///     }
/// }
/// fn main() {
///     let rsdr = TickTockReseeder { tick: true };
///
///     let inner = StdRng::new().unwrap();
///     let mut rng = ReseedingRng::new(inner, 10, rsdr);
///
///     // this will repeat, because it gets reseeded very regularly.
///     println!("{}", rng.gen_ascii_str(100));
/// }
///
/// ```
pub trait Reseeder<R> {
    /// Reseed the given RNG.
    fn reseed(&mut self, rng: &mut R);
}

/// Reseed an RNG using a `Default` instance. This reseeds by
/// replacing the RNG with the result of a `Default::default` call.
pub struct ReseedWithDefault;

impl<R: Rng + Default> Reseeder<R> for ReseedWithDefault {
    fn reseed(&mut self, rng: &mut R) {
        *rng = Default::default();
    }
}
impl Default for ReseedWithDefault {
    fn default() -> ReseedWithDefault { ReseedWithDefault }
}

#[cfg(test)]
mod test {
    use super::{ReseedingRng, ReseedWithDefault};
    use std::default::Default;
    use {SeedableRng, Rng};

    struct Counter {
        i: u32
    }

    impl Rng for Counter {
        fn next_u32(&mut self) -> u32 {
            self.i += 1;
            // very random
            self.i - 1
        }
    }
    impl Default for Counter {
        fn default() -> Counter {
            Counter { i: 0 }
        }
    }
    impl SeedableRng<u32> for Counter {
        fn reseed(&mut self, seed: u32) {
            self.i = seed;
        }
        fn from_seed(seed: u32) -> Counter {
            Counter { i: seed }
        }
    }
    type MyRng = ReseedingRng<Counter, ReseedWithDefault>;

    #[test]
    fn test_reseeding() {
        let mut rs = ReseedingRng::new(Counter {i:0}, 400, ReseedWithDefault);

        let mut i = 0;
        for _ in range(0, 1000) {
            assert_eq!(rs.next_u32(), i % 100);
            i += 1;
        }
    }

    #[test]
    fn test_rng_seeded() {
        let mut ra: MyRng = SeedableRng::from_seed((ReseedWithDefault, 2));
        let mut rb: MyRng = SeedableRng::from_seed((ReseedWithDefault, 2));
        assert_eq!(ra.gen_ascii_str(100u), rb.gen_ascii_str(100u));
    }

    #[test]
    fn test_rng_reseed() {
        let mut r: MyRng = SeedableRng::from_seed((ReseedWithDefault, 3));
        let string1 = r.gen_ascii_str(100);

        r.reseed((ReseedWithDefault, 3));

        let string2 = r.gen_ascii_str(100);
        assert_eq!(string1, string2);
    }

    static fill_bytes_v_len: uint = 13579;
    #[test]
    fn test_rng_fill_bytes() {
        use task_rng;
        let mut v = ~[0u8, .. fill_bytes_v_len];
        task_rng().fill_bytes(v);

        // Sanity test: if we've gotten here, `fill_bytes` has not infinitely
        // recursed.
        assert_eq!(v.len(), fill_bytes_v_len);

        // To test that `fill_bytes` actually did something, check that the
        // average of `v` is not 0.
        let mut sum = 0.0;
        for &x in v.iter() {
            sum += x as f64;
        }
        assert!(sum / v.len() as f64 != 0.0);
    }
}
