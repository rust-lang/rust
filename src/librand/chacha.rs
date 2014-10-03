// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! The ChaCha random number generator.

use core::prelude::*;

use {Rng, SeedableRng, Rand};

static KEY_WORDS    : uint =  8; // 8 words for the 256-bit key
static STATE_WORDS  : uint = 16;
static CHACHA_ROUNDS: uint = 20; // Cryptographically secure from 8 upwards as of this writing

/// A random number generator that uses the ChaCha20 algorithm [1].
///
/// The ChaCha algorithm is widely accepted as suitable for
/// cryptographic purposes, but this implementation has not been
/// verified as such. Prefer a generator like `OsRng` that defers to
/// the operating system for cases that need high security.
///
/// [1]: D. J. Bernstein, [*ChaCha, a variant of
/// Salsa20*](http://cr.yp.to/chacha.html)

pub struct ChaChaRng {
    buffer:  [u32, ..STATE_WORDS], // Internal buffer of output
    state:   [u32, ..STATE_WORDS], // Initial state
    index:   uint,                 // Index into state
}

static EMPTY: ChaChaRng = ChaChaRng {
    buffer:  [0, ..STATE_WORDS],
    state:   [0, ..STATE_WORDS],
    index:   STATE_WORDS
};


macro_rules! quarter_round{
    ($a: expr, $b: expr, $c: expr, $d: expr) => {{
        $a += $b; $d ^= $a; $d = $d.rotate_left(16);
        $c += $d; $b ^= $c; $b = $b.rotate_left(12);
        $a += $b; $d ^= $a; $d = $d.rotate_left( 8);
        $c += $d; $b ^= $c; $b = $b.rotate_left( 7);
    }}
}

macro_rules! double_round{
    ($x: expr) => {{
        // Column round
        quarter_round!($x[ 0], $x[ 4], $x[ 8], $x[12]);
        quarter_round!($x[ 1], $x[ 5], $x[ 9], $x[13]);
        quarter_round!($x[ 2], $x[ 6], $x[10], $x[14]);
        quarter_round!($x[ 3], $x[ 7], $x[11], $x[15]);
        // Diagonal round
        quarter_round!($x[ 0], $x[ 5], $x[10], $x[15]);
        quarter_round!($x[ 1], $x[ 6], $x[11], $x[12]);
        quarter_round!($x[ 2], $x[ 7], $x[ 8], $x[13]);
        quarter_round!($x[ 3], $x[ 4], $x[ 9], $x[14]);
    }}
}

#[inline]
fn core(output: &mut [u32, ..STATE_WORDS], input: &[u32, ..STATE_WORDS]) {
    *output = *input;

    for _ in range(0, CHACHA_ROUNDS / 2) {
        double_round!(output);
    }

    for i in range(0, STATE_WORDS) {
        output[i] += input[i];
    }
}

impl ChaChaRng {

    /// Create an ChaCha random number generator using the default
    /// fixed key of 8 zero words.
    pub fn new_unseeded() -> ChaChaRng {
        let mut rng = EMPTY;
        rng.init(&[0, ..KEY_WORDS]);
        rng
    }

    /// Sets the internal 128-bit ChaCha counter to
    /// a user-provided value. This permits jumping
    /// arbitrarily ahead (or backwards) in the pseudorandom stream.
    ///
    /// Since the nonce words are used to extend the counter to 128 bits,
    /// users wishing to obtain the conventional ChaCha pseudorandom stream
    /// associated with a particular nonce can call this function with
    /// arguments `0, desired_nonce`.
    pub fn set_counter(&mut self, counter_low: u64, counter_high: u64) {
        self.state[12] = (counter_low >>  0) as u32;
        self.state[13] = (counter_low >> 32) as u32;
        self.state[14] = (counter_high >>  0) as u32;
        self.state[15] = (counter_high >> 32) as u32;
        self.index = STATE_WORDS; // force recomputation
    }

    /// Initializes `self.state` with the appropriate key and constants
    ///
    /// We deviate slightly from the ChaCha specification regarding
    /// the nonce, which is used to extend the counter to 128 bits.
    /// This is provably as strong as the original cipher, though,
    /// since any distinguishing attack on our variant also works
    /// against ChaCha with a chosen-nonce. See the XSalsa20 [1]
    /// security proof for a more involved example of this.
    ///
    /// The modified word layout is:
    /// ```notrust
    /// constant constant constant constant
    /// key      key      key      key
    /// key      key      key      key
    /// counter  counter  counter  counter
    /// ```
    /// [1]: Daniel J. Bernstein. [*Extending the Salsa20
    /// nonce.*](http://cr.yp.to/papers.html#xsalsa)
    fn init(&mut self, key: &[u32, ..KEY_WORDS]) {
        self.state[0] = 0x61707865;
        self.state[1] = 0x3320646E;
        self.state[2] = 0x79622D32;
        self.state[3] = 0x6B206574;

        for i in range(0, KEY_WORDS) {
            self.state[4+i] = key[i];
        }

        self.state[12] = 0;
        self.state[13] = 0;
        self.state[14] = 0;
        self.state[15] = 0;

        self.index = STATE_WORDS;
    }

    /// Refill the internal output buffer (`self.buffer`)
    fn update(&mut self) {
        core(&mut self.buffer, &self.state);
        self.index = 0;
        // update 128-bit counter
        self.state[12] += 1;
        if self.state[12] != 0 { return };
        self.state[13] += 1;
        if self.state[13] != 0 { return };
        self.state[14] += 1;
        if self.state[14] != 0 { return };
        self.state[15] += 1;
    }
}

impl Rng for ChaChaRng {
    #[inline]
    fn next_u32(&mut self) -> u32 {
        if self.index == STATE_WORDS {
            self.update();
        }

        let value = self.buffer[self.index % STATE_WORDS];
        self.index += 1;
        value
    }
}

impl<'a> SeedableRng<&'a [u32]> for ChaChaRng {

    fn reseed(&mut self, seed: &'a [u32]) {
        // reset state
        self.init(&[0u32, ..KEY_WORDS]);
        // set key inplace
        let key = self.state.slice_mut(4, 4+KEY_WORDS);
        for (k, s) in key.iter_mut().zip(seed.iter()) {
            *k = *s;
        }
    }

    /// Create a ChaCha generator from a seed,
    /// obtained from a variable-length u32 array.
    /// Only up to 8 words are used; if less than 8
    /// words are used, the remaining are set to zero.
    fn from_seed(seed: &'a [u32]) -> ChaChaRng {
        let mut rng = EMPTY;
        rng.reseed(seed);
        rng
    }
}

impl Rand for ChaChaRng {
    fn rand<R: Rng>(other: &mut R) -> ChaChaRng {
        let mut key : [u32, ..KEY_WORDS] = [0, ..KEY_WORDS];
        for word in key.iter_mut() {
            *word = other.gen();
        }
        SeedableRng::from_seed(key.as_slice())
    }
}


#[cfg(test)]
mod test {
    use std::prelude::*;

    use core::iter::order;
    use {Rng, SeedableRng};
    use super::ChaChaRng;

    #[test]
    fn test_rng_rand_seeded() {
        let s = ::test::rng().gen_iter::<u32>().take(8).collect::<Vec<u32>>();
        let mut ra: ChaChaRng = SeedableRng::from_seed(s.as_slice());
        let mut rb: ChaChaRng = SeedableRng::from_seed(s.as_slice());
        assert!(order::equals(ra.gen_ascii_chars().take(100),
                              rb.gen_ascii_chars().take(100)));
    }

    #[test]
    fn test_rng_seeded() {
        let seed : &[_] = &[0,1,2,3,4,5,6,7];
        let mut ra: ChaChaRng = SeedableRng::from_seed(seed);
        let mut rb: ChaChaRng = SeedableRng::from_seed(seed);
        assert!(order::equals(ra.gen_ascii_chars().take(100),
                              rb.gen_ascii_chars().take(100)));
    }

    #[test]
    fn test_rng_reseed() {
        let s = ::test::rng().gen_iter::<u32>().take(8).collect::<Vec<u32>>();
        let mut r: ChaChaRng = SeedableRng::from_seed(s.as_slice());
        let string1: String = r.gen_ascii_chars().take(100).collect();

        r.reseed(s.as_slice());

        let string2: String = r.gen_ascii_chars().take(100).collect();
        assert_eq!(string1, string2);
    }

    #[test]
    fn test_rng_true_values() {
        // Test vectors 1 and 2 from
        // http://tools.ietf.org/html/draft-nir-cfrg-chacha20-poly1305-04
        let seed : &[_] = &[0u32, ..8];
        let mut ra: ChaChaRng = SeedableRng::from_seed(seed);

        let v = Vec::from_fn(16, |_| ra.next_u32());
        assert_eq!(v,
                   vec!(0xade0b876, 0x903df1a0, 0xe56a5d40, 0x28bd8653,
                        0xb819d2bd, 0x1aed8da0, 0xccef36a8, 0xc70d778b,
                        0x7c5941da, 0x8d485751, 0x3fe02477, 0x374ad8b8,
                        0xf4b8436a, 0x1ca11815, 0x69b687c3, 0x8665eeb2));

        let v = Vec::from_fn(16, |_| ra.next_u32());
        assert_eq!(v,
                   vec!(0xbee7079f, 0x7a385155, 0x7c97ba98, 0x0d082d73,
                        0xa0290fcb, 0x6965e348, 0x3e53c612, 0xed7aee32,
                        0x7621b729, 0x434ee69c, 0xb03371d5, 0xd539d874,
                        0x281fed31, 0x45fb0a51, 0x1f0ae1ac, 0x6f4d794b));


        let seed : &[_] = &[0,1,2,3,4,5,6,7];
        let mut ra: ChaChaRng = SeedableRng::from_seed(seed);

        // Store the 17*i-th 32-bit word,
        // i.e., the i-th word of the i-th 16-word block
        let mut v : Vec<u32> = Vec::new();
        for _ in range(0u, 16) {
            v.push(ra.next_u32());
            for _ in range(0u, 16) {
                ra.next_u32();
            }
        }

        assert_eq!(v,
                   vec!(0xf225c81a, 0x6ab1be57, 0x04d42951, 0x70858036,
                        0x49884684, 0x64efec72, 0x4be2d186, 0x3615b384,
                        0x11cfa18e, 0xd3c50049, 0x75c775f6, 0x434c6530,
                        0x2c5bad8f, 0x898881dc, 0x5f1c86d9, 0xc1f8e7f4));
    }
}

