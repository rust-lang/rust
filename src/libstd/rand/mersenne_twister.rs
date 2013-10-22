// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Implementations of the 32- and 64-bit variants of the Mersenne
//! Twister random number generator.
use cmp;
use vec::raw;
use mem;
use iter::range;
use option::{Some, None};
use rand::{Rng, SeedableRng, OSRng};

static MT_N: uint = 624;
static MT_M: uint = 397;
static MT_A: u32 = 0x9908b0df;
static MT_HI: u32 = 0x8000_0000;
static MT_LO: u32 = 0x7fff_ffff;

/// A random number generator that uses the Mersenne Twister
/// algorithm[1].
///
/// This is not a cryptographically secure RNG.
///
/// [1]: Matsumoto, M.; Nishimura, T. (1998). "Mersenne twister: a
/// 623-dimensionally equidistributed uniform pseudo-random number
/// generator". *ACM Transactions on Modeling and Computer Simulation
/// 8* (1):
/// 3–30. doi:[10.1145/272991.272995](http://dx.doi.org/10.1145%2F272991.272995)
pub struct MT19937Rng {
    priv state: [u32, .. MT_N],
    priv index: uint
}

impl MT19937Rng {
    /// Create an Mersenne Twister random number generator with a
    /// random seed.
    pub fn new() -> MT19937Rng {
        let mut seed = [0u32, .. MT_N];
        unsafe {
            let ptr = raw::to_mut_ptr(seed);

            do raw::mut_buf_as_slice(ptr as *mut u8, mem::size_of_val(&seed)) |slice| {
                OSRng::new().fill_bytes(slice);
            }
        }
        SeedableRng::from_seed(seed.as_slice())
    }

    #[inline]
    fn generate_numbers(&mut self) {
        macro_rules! MIXBITS ( ($u:expr, $v:expr) => { (($u & MT_HI) | ($v & MT_LO)) } );
        macro_rules! TWIST (
            ($u:expr, $v:expr) => { ((MIXBITS!($u, $v) >> 1) ^ (($v & 1) * MT_A)) }
        );

        // unsafe pointers to match the C implementation.
        let mut p = raw::to_mut_ptr(self.state);
        unsafe {
            for _ in range(0, MT_N - MT_M) {
                *p = *p.offset(MT_M as int) ^ TWIST!(*p, *p.offset(1));
                p = p.offset(1);
            }
            for _ in range(MT_N - MT_M, MT_N - 1) {
                *p = *p.offset(MT_M as int - MT_N as int) ^ TWIST!(*p, *p.offset(1));
                p = p.offset(1);
            }
            *p = *p.offset(MT_M as int - MT_N as int) ^ TWIST!(*p, self.state[0]);
        }
        self.index = 0;
    }
}

impl Rng for MT19937Rng {
    #[inline]
    fn next_u32(&mut self) -> u32 {
        if self.index >= MT_N {
            self.generate_numbers();
        }

        let mut y = unsafe { self.state.unsafe_get(self.index) };
        self.index += 1;

        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c5680;
        y ^= (y << 15) & 0xefc60000;
        y ^ (y >> 18)
    }
}

// This is ugly, but is it's necessary to be able to work directly
// with SeedableRng: the seeding procedure for MT19937 differs for a
// single integer and a vector.
trait MT19937RngSeed { fn reseed(&self, &mut MT19937Rng); }
impl MT19937RngSeed for u32 {
    fn reseed(&self, rng: &mut MT19937Rng) {
        rng.state[0] = *self;
        for i in range(1, MT_N) {
            rng.state[i] = 1812433253 * (rng.state[i-1] ^ (rng.state[i-1] >> 30)) + i as u32;
        }

        rng.index = MT_N;
    }
}
impl<'self> MT19937RngSeed for &'self [u32] {
    fn reseed(&self, rng: &mut MT19937Rng) {
        rng.reseed(19650218u32);

        let len = self.len();
        let lim = cmp::max(len, MT_N);

        let mut i = 1;
        let mut j = 0;
        for _ in range(0, lim) {
            let val = (rng.state[i] ^
                       (1664525 * (rng.state[i-1] ^ (rng.state[i-1] >> 30)))) + (*self)[j] + j;
            rng.state[i] = val;

            i += 1;
            j += 1;

            if (i >= MT_N) { rng.state[0] = rng.state[MT_N - 1]; i = 1; }
            if (j as uint >= len) { j = 0; }
        }

        for _ in range(0, MT_N - 1) {
            let val = (rng.state[i] ^
                       (1566083941 * (rng.state[i-1] ^ (rng.state[i-1] >> 30)))) - i as u32;
            rng.state[i] = val;
            i += 1;
            if (i >= MT_N) { rng.state[0] = rng.state[MT_N - 1]; i = 1; }
        }

        rng.state[0] = 0x8000_0000;
        rng.index = MT_N;
    }
}
impl<Seed: MT19937RngSeed> SeedableRng<Seed> for MT19937Rng {
    fn reseed(&mut self, seed: Seed) {
        seed.reseed(self)
    }
    fn from_seed(seed: Seed) -> MT19937Rng {
        let mut r = MT19937Rng { state: [0, .. MT_N], index: 0 };
        r.reseed(seed);
        r
    }
}

static MT64_N: uint = 312;
static MT64_M: uint = 156;
static MT64_A: u64 = 0xB502_6F5A_A966_19E9;
static MT64_HI: u64 = 0xffff_ffff_8000_0000;
static MT64_LO: u64 = 0x0000_0000_7fff_ffff;

/// A random number generator that uses the 64 bit version of the
/// Mersenne Twister algorithm[1].
///
/// This is not a cryptographically secure RNG.
///
/// [1]: Matsumoto, M. [*Mersenne Twister 64bit
/// version*](http://www.math.sci.hiroshima-u.ac.jp/~m-mat/MT/emt64.html)
#[allow(non_camel_case_types)] // calling this `MT1993764Rng` would be bad.
pub struct MT19937_64Rng {
    priv state: [u64, .. MT64_N],
    priv index: uint
}

impl MT19937_64Rng {
    /// Create an 64-bit Mersenne Twister random number generator with
    /// a random seed.
    pub fn new() -> MT19937_64Rng {
        let mut seed = [0u64, .. MT_N];
        unsafe {
            let ptr = raw::to_mut_ptr(seed);

            do raw::mut_buf_as_slice(ptr as *mut u8, mem::size_of_val(&seed)) |slice| {
                OSRng::new().fill_bytes(slice);
            }
        }
        SeedableRng::from_seed(seed.as_slice())
    }
    #[inline]
    fn generate_numbers(&mut self) {
        for i in range(0, MT64_N - MT64_M) {
            let x = (self.state[i] & MT64_HI) | (self.state[i + 1] & MT64_LO);
            self.state[i] = self.state[i + MT64_M] ^ (x >> 1) ^ ((x & 1) * MT64_A);
        }
        for i in range(MT64_N - MT64_M, MT64_N - 1) {
            let x = (self.state[i] & MT64_HI) | (self.state[i + 1] & MT64_LO);
            self.state[i] = self.state[i + MT64_M - MT64_N] ^ (x >> 1) ^ ((x & 1) * MT64_A);
        }
        let x = (self.state[MT64_N - 1] & MT64_HI) | (self.state[0] & MT64_LO);
        self.state[MT64_N - 1] = self.state[MT64_M - 1] ^ (x >> 1) ^ ((x & 1) * MT64_A);

        self.index = 0;
    }
}

impl Rng for MT19937_64Rng {
    #[inline]
    fn next_u32(&mut self) -> u32 { self.next_u64() as u32 }

    #[inline]
    fn next_u64(&mut self) -> u64 {
        if self.index >= MT64_N {
            self.generate_numbers();
        }

        let mut x = self.state[self.index];
        self.index += 1;
        x ^= (x >> 29) & 0x5555555555555555;
        x ^= (x << 17) & 0x71D67FFFEDA60000;
        x ^= (x << 37) & 0xFFF7EEE000000000;
        x ^ (x >> 43)
    }
}

#[allow(non_camel_case_types)] // removing the _ would be silly
trait MT19937_64RngSeed {
    fn reseed(&self, &mut MT19937_64Rng);
}
impl MT19937_64RngSeed for u64 {
    fn reseed(&self, rng: &mut MT19937_64Rng) {
        rng.state[0] = *self;
        for i in range(1, MT64_N) {
            rng.state[i] = 6364136223846793005 *
                (rng.state[i-1] ^ (rng.state[i-1] >> 62)) + i as u64;
        }

        rng.index = MT64_N;
    }
}
impl<'self> MT19937_64RngSeed for &'self [u64] {
    fn reseed(&self, rng: &mut MT19937_64Rng) {
        rng.reseed(19650218u64);

        let len = self.len();
        let lim = cmp::max(len, MT64_N);
        let mut i = 1;
        let mut j = 0;
        for _ in range(0, lim) {
            let val = (rng.state[i] ^
                       (3935559000370003845 * (rng.state[i-1] ^ (rng.state[i-1] >> 62)))) +
                (*self)[j] + j;
            rng.state[i] = val;

            i += 1;
            j += 1;

            if (i >= MT64_N) { rng.state[0] = rng.state[MT64_N - 1]; i = 1; }
            if (j as uint >= len) { j = 0; }
        }

        for _ in range(0, MT64_N - 1) {
            rng.state[i] = (rng.state[i] ^
                             (2862933555777941757 * (rng.state[i-1] ^ (rng.state[i-1] >> 62))))
                - i as u64;

            i += 1;
            if (i >= MT64_N) { rng.state[0] = rng.state[MT64_N - 1]; i = 1; }
        }
        rng.state[0] = 1 << 63;
        rng.index = MT64_N;
    }
}

impl<Seed: MT19937_64RngSeed> SeedableRng<Seed> for MT19937_64Rng {
    fn reseed(&mut self, seed: Seed) {
        seed.reseed(self)
    }
    fn from_seed(seed: Seed) -> MT19937_64Rng {
        let mut r = MT19937_64Rng { state: [0, .. MT64_N], index: 0 };
        r.reseed(seed);
        r
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use rand::{OSRng, SeedableRng};
    use vec;
    use iter::range;
    use option::{Some, None};

    #[test]
    fn test_rng_32_rand_seeded() {
        let s = OSRng::new().gen_vec::<u32>(256);
        let mut ra: MT19937Rng = SeedableRng::from_seed(s.as_slice());
        let mut rb: MT19937Rng = SeedableRng::from_seed(s.as_slice());
        assert_eq!(ra.gen_ascii_str(100u), rb.gen_ascii_str(100u));
    }
    #[test]
    fn test_rng_64_rand_seeded() {
        let s = OSRng::new().gen_vec::<u64>(256);
        let mut ra: MT19937_64Rng = SeedableRng::from_seed(s.as_slice());
        let mut rb: MT19937_64Rng = SeedableRng::from_seed(s.as_slice());
        assert_eq!(ra.gen_ascii_str(100u), rb.gen_ascii_str(100u));
    }

    #[test]
    fn test_rng_32_seeded() {
        let seed = &[1u32, 23, 456, 7890, 12345];
        let mut ra: MT19937Rng = SeedableRng::from_seed(seed);
        let mut rb: MT19937Rng = SeedableRng::from_seed(seed);
        assert_eq!(ra.gen_ascii_str(100u), rb.gen_ascii_str(100u));
    }
    #[test]
    fn test_rng_64_seeded() {
        let seed = &[1u64, 23, 456, 7890, 12345];
        let mut ra: MT19937_64Rng = SeedableRng::from_seed(seed);
        let mut rb: MT19937_64Rng = SeedableRng::from_seed(seed);
        assert_eq!(ra.gen_ascii_str(100u), rb.gen_ascii_str(100u));
    }

    #[test]
    fn test_rng_32_reseed() {
        let s = OSRng::new().gen_vec::<u32>(256);
        let mut r: MT19937Rng = SeedableRng::from_seed(s.as_slice());
        let string1 = r.gen_ascii_str(100);

        r.reseed(s.as_slice());

        let string2 = r.gen_ascii_str(100);
        assert_eq!(string1, string2);
    }
    #[test]
    fn test_rng_64_reseed() {
        let s = OSRng::new().gen_vec::<u64>(256);
        let mut r: MT19937_64Rng = SeedableRng::from_seed(s.as_slice());
        let string1 = r.gen_ascii_str(100);

        r.reseed(s.as_slice());

        let string2 = r.gen_ascii_str(100);
        assert_eq!(string1, string2);
    }

    #[test]
    fn test_rng_32_true_values() {
        // real values taken from
        // http://www.math.sci.hiroshima-u.ac.jp/~m-mat/MT/MT2002/emt19937ar.html

        let mut ra: MT19937Rng = SeedableRng::from_seed(1234567890u32);
        let v = vec::from_fn(10, |_| ra.next_u32());
        assert_eq!(v,
                   ~[2657703298, 1462474751, 2541004134, 640082991, 3816866956,
                     998313779, 3829628193, 1854614443, 1965237353, 3628085564]);

        let seed: &[u32] = &[12345, 67890, 54321, 9876];
        let mut rb: MT19937Rng = SeedableRng::from_seed(seed);
        // skip forward to the 10000th number
        for _ in range(0, 10000) { rb.next_u32(); }

        let v = vec::from_fn(10, |_| rb.next_u32());
        assert_eq!(v,
                   ~[895988861, 1625433920, 904396948, 682323433, 1842852260,
                     3882768219, 1125692803, 4154813681, 3323837245, 2988885577]);
    }
    #[test]
    fn test_rng_64_true_values() {
        // real values taken from
        // http://www.math.sci.hiroshima-u.ac.jp/~m-mat/MT/emt64.html

        let mut ra: MT19937_64Rng = SeedableRng::from_seed(1234567890u64);
        let v = vec::from_fn(10, |_| ra.next_u64());
        assert_eq!(v,
                   ~[17945193607607270098, 13741097643249962112, 11301849514948974241,
                     9366053755680083846, 1706575204080707832, 11158352244489866165,
                     13009036798513897099, 4733827879644583927, 8074293671969069159,
                     7781079419436238939]);

        let seed: &[u64] = &[12345, 67890, 54321, 9876];
        let mut rb: MT19937_64Rng = SeedableRng::from_seed(seed);
        // skip forward to the 10000th number
        for _ in range(0, 10000) { rb.next_u64(); }

        let v = vec::from_fn(10, |_| rb.next_u64());
        assert_eq!(v,
                   ~[4917329019666999141, 8976322753452851857, 11836095949751687898,
                     8824236106928878382, 1760621375340792993, 12993611849689829524,
                     2697601593788585026, 12456531818874198756, 7206325672628826773,
                     15989680927513200155]);
    }

}
