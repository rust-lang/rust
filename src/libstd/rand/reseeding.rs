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

use rand::{Rng, SeedableRng};
use default::Default;

/// How many bytes of entropy the underling RNG is allowed to generate
/// before it is reseeded.
static DEFAULT_GENERATION_THRESHOLD: uint = 32 * 1024;

/// A wrapper around any RNG which reseeds the underlying RNG after it
/// has generated a certain number of random bytes.
pub struct ReseedingRng<R, Rsdr> {
    priv rng: R,
    priv generation_threshold: uint,
    priv bytes_generated: uint,
    /// Controls the behaviour when reseeding the RNG.
    reseeder: Rsdr
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
        self.fill_bytes(dest)
    }
}

impl<S, R: SeedableRng<S>, Rsdr: Reseeder<R> + Default> SeedableRng<S> for ReseedingRng<R, Rsdr> {
    fn reseed(&mut self, seed: S) {
        self.rng.reseed(seed);
        self.bytes_generated = 0;
    }
    /// Create a new `ReseedingRng` from the given seed. This uses
    /// default values for both `generation_threshold` and `reseeder`.
    fn from_seed(seed: S) -> ReseedingRng<R, Rsdr> {
        ReseedingRng {
            rng: SeedableRng::from_seed(seed),
            generation_threshold: DEFAULT_GENERATION_THRESHOLD,
            bytes_generated: 0,
            reseeder: Default::default()
        }
    }
}

/// Something that can be used to reseed an RNG via `ReseedingRng`.
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

#[cfg(test)]
mod test {
    use super::*;
    use rand::Rng;
    use default::Default;
    use iter::range;
    use option::{None, Some};

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

    #[test]
    fn test_reseeding() {
        let mut rs = ReseedingRng::new(Counter {i:0}, 400, ReseedWithDefault);

        let mut i = 0;
        for _ in range(0, 1000) {
            assert_eq!(rs.next_u32(), i % 100);
            i += 1;
        }
    }
}
