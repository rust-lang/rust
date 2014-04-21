// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! The ISAAC random number generator.

use core::prelude::*;
use core::iter::{range_step, Repeat};
use core::slice::raw;
use core::mem;

use {Rng, SeedableRng, Rand};

static RAND_SIZE_LEN: u32 = 8;
static RAND_SIZE: u32 = 1 << (RAND_SIZE_LEN as uint);
static RAND_SIZE_UINT: uint = 1 << (RAND_SIZE_LEN as uint);

/// A random number generator that uses the ISAAC algorithm[1].
///
/// The ISAAC algorithm is generally accepted as suitable for
/// cryptographic purposes, but this implementation has not be
/// verified as such. Prefer a generator like `OsRng` that defers to
/// the operating system for cases that need high security.
///
/// [1]: Bob Jenkins, [*ISAAC: A fast cryptographic random number
/// generator*](http://www.burtleburtle.net/bob/rand/isaacafa.html)
pub struct IsaacRng {
    cnt: u32,
    rsl: [u32, ..RAND_SIZE_UINT],
    mem: [u32, ..RAND_SIZE_UINT],
    a: u32,
    b: u32,
    c: u32
}
static EMPTY: IsaacRng = IsaacRng {
    cnt: 0,
    rsl: [0, ..RAND_SIZE_UINT],
    mem: [0, ..RAND_SIZE_UINT],
    a: 0, b: 0, c: 0
};

impl IsaacRng {
    /// Create an ISAAC random number generator using the default
    /// fixed seed.
    pub fn new_unseeded() -> IsaacRng {
        let mut rng = EMPTY;
        rng.init(false);
        rng
    }

    /// Initialises `self`. If `use_rsl` is true, then use the current value
    /// of `rsl` as a seed, otherwise construct one algorithmically (not
    /// randomly).
    fn init(&mut self, use_rsl: bool) {
        let mut a = 0x9e3779b9;
        let mut b = a;
        let mut c = a;
        let mut d = a;
        let mut e = a;
        let mut f = a;
        let mut g = a;
        let mut h = a;

        macro_rules! mix(
            () => {{
                a^=b<<11; d+=a; b+=c;
                b^=c>>2;  e+=b; c+=d;
                c^=d<<8;  f+=c; d+=e;
                d^=e>>16; g+=d; e+=f;
                e^=f<<10; h+=e; f+=g;
                f^=g>>4;  a+=f; g+=h;
                g^=h<<8;  b+=g; h+=a;
                h^=a>>9;  c+=h; a+=b;
            }}
        );

        for _ in range(0u, 4) {
            mix!();
        }

        if use_rsl {
            macro_rules! memloop (
                ($arr:expr) => {{
                    for i in range_step(0, RAND_SIZE as uint, 8) {
                        a+=$arr[i  ]; b+=$arr[i+1];
                        c+=$arr[i+2]; d+=$arr[i+3];
                        e+=$arr[i+4]; f+=$arr[i+5];
                        g+=$arr[i+6]; h+=$arr[i+7];
                        mix!();
                        self.mem[i  ]=a; self.mem[i+1]=b;
                        self.mem[i+2]=c; self.mem[i+3]=d;
                        self.mem[i+4]=e; self.mem[i+5]=f;
                        self.mem[i+6]=g; self.mem[i+7]=h;
                    }
                }}
            );

            memloop!(self.rsl);
            memloop!(self.mem);
        } else {
            for i in range_step(0, RAND_SIZE as uint, 8) {
                mix!();
                self.mem[i  ]=a; self.mem[i+1]=b;
                self.mem[i+2]=c; self.mem[i+3]=d;
                self.mem[i+4]=e; self.mem[i+5]=f;
                self.mem[i+6]=g; self.mem[i+7]=h;
            }
        }

        self.isaac();
    }

    /// Refills the output buffer (`self.rsl`)
    #[inline]
    #[allow(unsigned_negate)]
    fn isaac(&mut self) {
        self.c += 1;
        // abbreviations
        let mut a = self.a;
        let mut b = self.b + self.c;

        static MIDPOINT: uint = (RAND_SIZE / 2) as uint;

        macro_rules! ind (($x:expr) => {
            self.mem[(($x >> 2) as uint & ((RAND_SIZE - 1) as uint))]
        });
        macro_rules! rngstepp(
            ($j:expr, $shift:expr) => {{
                let base = $j;
                let mix = a << $shift as uint;

                let x = self.mem[base  + mr_offset];
                a = (a ^ mix) + self.mem[base + m2_offset];
                let y = ind!(x) + a + b;
                self.mem[base + mr_offset] = y;

                b = ind!(y >> RAND_SIZE_LEN as uint) + x;
                self.rsl[base + mr_offset] = b;
            }}
        );
        macro_rules! rngstepn(
            ($j:expr, $shift:expr) => {{
                let base = $j;
                let mix = a >> $shift as uint;

                let x = self.mem[base  + mr_offset];
                a = (a ^ mix) + self.mem[base + m2_offset];
                let y = ind!(x) + a + b;
                self.mem[base + mr_offset] = y;

                b = ind!(y >> RAND_SIZE_LEN as uint) + x;
                self.rsl[base + mr_offset] = b;
            }}
        );

        let r = [(0, MIDPOINT), (MIDPOINT, 0)];
        for &(mr_offset, m2_offset) in r.iter() {
            for i in range_step(0u, MIDPOINT, 4) {
                rngstepp!(i + 0, 13);
                rngstepn!(i + 1, 6);
                rngstepp!(i + 2, 2);
                rngstepn!(i + 3, 16);
            }
        }

        self.a = a;
        self.b = b;
        self.cnt = RAND_SIZE;
    }
}

impl Rng for IsaacRng {
    #[inline]
    fn next_u32(&mut self) -> u32 {
        if self.cnt == 0 {
            // make some more numbers
            self.isaac();
        }
        self.cnt -= 1;
        self.rsl[self.cnt as uint]
    }
}

impl<'a> SeedableRng<&'a [u32]> for IsaacRng {
    fn reseed(&mut self, seed: &'a [u32]) {
        // make the seed into [seed[0], seed[1], ..., seed[seed.len()
        // - 1], 0, 0, ...], to fill rng.rsl.
        let seed_iter = seed.iter().map(|&x| x).chain(Repeat::new(0u32));

        for (rsl_elem, seed_elem) in self.rsl.mut_iter().zip(seed_iter) {
            *rsl_elem = seed_elem;
        }
        self.cnt = 0;
        self.a = 0;
        self.b = 0;
        self.c = 0;

        self.init(true);
    }

    /// Create an ISAAC random number generator with a seed. This can
    /// be any length, although the maximum number of elements used is
    /// 256 and any more will be silently ignored. A generator
    /// constructed with a given seed will generate the same sequence
    /// of values as all other generators constructed with that seed.
    fn from_seed(seed: &'a [u32]) -> IsaacRng {
        let mut rng = EMPTY;
        rng.reseed(seed);
        rng
    }
}

impl Rand for IsaacRng {
    fn rand<R: Rng>(other: &mut R) -> IsaacRng {
        let mut ret = EMPTY;
        unsafe {
            let ptr = ret.rsl.as_mut_ptr();

            raw::mut_buf_as_slice(ptr as *mut u8,
                                  mem::size_of_val(&ret.rsl), |slice| {
                other.fill_bytes(slice);
            })
        }
        ret.cnt = 0;
        ret.a = 0;
        ret.b = 0;
        ret.c = 0;

        ret.init(true);
        return ret;
    }
}

static RAND_SIZE_64_LEN: uint = 8;
static RAND_SIZE_64: uint = 1 << RAND_SIZE_64_LEN;

/// A random number generator that uses ISAAC-64[1], the 64-bit
/// variant of the ISAAC algorithm.
///
/// The ISAAC algorithm is generally accepted as suitable for
/// cryptographic purposes, but this implementation has not be
/// verified as such. Prefer a generator like `OsRng` that defers to
/// the operating system for cases that need high security.
///
/// [1]: Bob Jenkins, [*ISAAC: A fast cryptographic random number
/// generator*](http://www.burtleburtle.net/bob/rand/isaacafa.html)
pub struct Isaac64Rng {
    cnt: uint,
    rsl: [u64, .. RAND_SIZE_64],
    mem: [u64, .. RAND_SIZE_64],
    a: u64,
    b: u64,
    c: u64,
}

static EMPTY_64: Isaac64Rng = Isaac64Rng {
    cnt: 0,
    rsl: [0, .. RAND_SIZE_64],
    mem: [0, .. RAND_SIZE_64],
    a: 0, b: 0, c: 0,
};

impl Isaac64Rng {
    /// Create a 64-bit ISAAC random number generator using the
    /// default fixed seed.
    pub fn new_unseeded() -> Isaac64Rng {
        let mut rng = EMPTY_64;
        rng.init(false);
        rng
    }

    /// Initialises `self`. If `use_rsl` is true, then use the current value
    /// of `rsl` as a seed, otherwise construct one algorithmically (not
    /// randomly).
    fn init(&mut self, use_rsl: bool) {
        macro_rules! init (
            ($var:ident) => (
                let mut $var = 0x9e3779b97f4a7c13;
            )
        );
        init!(a); init!(b); init!(c); init!(d);
        init!(e); init!(f); init!(g); init!(h);

        macro_rules! mix(
            () => {{
                a-=e; f^=h>>9;  h+=a;
                b-=f; g^=a<<9;  a+=b;
                c-=g; h^=b>>23; b+=c;
                d-=h; a^=c<<15; c+=d;
                e-=a; b^=d>>14; d+=e;
                f-=b; c^=e<<20; e+=f;
                g-=c; d^=f>>17; f+=g;
                h-=d; e^=g<<14; g+=h;
            }}
        );

        for _ in range(0u, 4) {
            mix!();
        }

        if use_rsl {
            macro_rules! memloop (
                ($arr:expr) => {{
                    for i in range(0, RAND_SIZE_64 / 8).map(|i| i * 8) {
                        a+=$arr[i  ]; b+=$arr[i+1];
                        c+=$arr[i+2]; d+=$arr[i+3];
                        e+=$arr[i+4]; f+=$arr[i+5];
                        g+=$arr[i+6]; h+=$arr[i+7];
                        mix!();
                        self.mem[i  ]=a; self.mem[i+1]=b;
                        self.mem[i+2]=c; self.mem[i+3]=d;
                        self.mem[i+4]=e; self.mem[i+5]=f;
                        self.mem[i+6]=g; self.mem[i+7]=h;
                    }
                }}
            );

            memloop!(self.rsl);
            memloop!(self.mem);
        } else {
            for i in range(0, RAND_SIZE_64 / 8).map(|i| i * 8) {
                mix!();
                self.mem[i  ]=a; self.mem[i+1]=b;
                self.mem[i+2]=c; self.mem[i+3]=d;
                self.mem[i+4]=e; self.mem[i+5]=f;
                self.mem[i+6]=g; self.mem[i+7]=h;
            }
        }

        self.isaac64();
    }

    /// Refills the output buffer (`self.rsl`)
    fn isaac64(&mut self) {
        self.c += 1;
        // abbreviations
        let mut a = self.a;
        let mut b = self.b + self.c;
        static MIDPOINT: uint =  RAND_SIZE_64 / 2;
        static MP_VEC: [(uint, uint), .. 2] = [(0,MIDPOINT), (MIDPOINT, 0)];
        macro_rules! ind (
            ($x:expr) => {
                *self.mem.unsafe_ref(($x as uint >> 3) & (RAND_SIZE_64 - 1))
            }
        );
        macro_rules! rngstepp(
            ($j:expr, $shift:expr) => {{
                let base = base + $j;
                let mix = a ^ (a << $shift as uint);
                let mix = if $j == 0 {!mix} else {mix};

                unsafe {
                    let x = *self.mem.unsafe_ref(base + mr_offset);
                    a = mix + *self.mem.unsafe_ref(base + m2_offset);
                    let y = ind!(x) + a + b;
                    self.mem.unsafe_set(base + mr_offset, y);

                    b = ind!(y >> RAND_SIZE_64_LEN) + x;
                    self.rsl.unsafe_set(base + mr_offset, b);
                }
            }}
        );
        macro_rules! rngstepn(
            ($j:expr, $shift:expr) => {{
                let base = base + $j;
                let mix = a ^ (a >> $shift as uint);
                let mix = if $j == 0 {!mix} else {mix};

                unsafe {
                    let x = *self.mem.unsafe_ref(base + mr_offset);
                    a = mix + *self.mem.unsafe_ref(base + m2_offset);
                    let y = ind!(x) + a + b;
                    self.mem.unsafe_set(base + mr_offset, y);

                    b = ind!(y >> RAND_SIZE_64_LEN) + x;
                    self.rsl.unsafe_set(base + mr_offset, b);
                }
            }}
        );

        for &(mr_offset, m2_offset) in MP_VEC.iter() {
            for base in range(0, MIDPOINT / 4).map(|i| i * 4) {
                rngstepp!(0, 21);
                rngstepn!(1, 5);
                rngstepp!(2, 12);
                rngstepn!(3, 33);
            }
        }

        self.a = a;
        self.b = b;
        self.cnt = RAND_SIZE_64;
    }
}

impl Rng for Isaac64Rng {
    // FIXME #7771: having next_u32 like this should be unnecessary
    #[inline]
    fn next_u32(&mut self) -> u32 {
        self.next_u64() as u32
    }

    #[inline]
    fn next_u64(&mut self) -> u64 {
        if self.cnt == 0 {
            // make some more numbers
            self.isaac64();
        }
        self.cnt -= 1;
        unsafe { *self.rsl.unsafe_ref(self.cnt) }
    }
}

impl<'a> SeedableRng<&'a [u64]> for Isaac64Rng {
    fn reseed(&mut self, seed: &'a [u64]) {
        // make the seed into [seed[0], seed[1], ..., seed[seed.len()
        // - 1], 0, 0, ...], to fill rng.rsl.
        let seed_iter = seed.iter().map(|&x| x).chain(Repeat::new(0u64));

        for (rsl_elem, seed_elem) in self.rsl.mut_iter().zip(seed_iter) {
            *rsl_elem = seed_elem;
        }
        self.cnt = 0;
        self.a = 0;
        self.b = 0;
        self.c = 0;

        self.init(true);
    }

    /// Create an ISAAC random number generator with a seed. This can
    /// be any length, although the maximum number of elements used is
    /// 256 and any more will be silently ignored. A generator
    /// constructed with a given seed will generate the same sequence
    /// of values as all other generators constructed with that seed.
    fn from_seed(seed: &'a [u64]) -> Isaac64Rng {
        let mut rng = EMPTY_64;
        rng.reseed(seed);
        rng
    }
}

impl Rand for Isaac64Rng {
    fn rand<R: Rng>(other: &mut R) -> Isaac64Rng {
        let mut ret = EMPTY_64;
        unsafe {
            let ptr = ret.rsl.as_mut_ptr();

            raw::mut_buf_as_slice(ptr as *mut u8,
                                  mem::size_of_val(&ret.rsl), |slice| {
                other.fill_bytes(slice);
            })
        }
        ret.cnt = 0;
        ret.a = 0;
        ret.b = 0;
        ret.c = 0;

        ret.init(true);
        return ret;
    }
}

#[cfg(test)]
mod test {
    use std::prelude::*;

    use core::iter::order;
    use {Rng, SeedableRng};
    use super::{IsaacRng, Isaac64Rng};

    #[test]
    fn test_rng_32_rand_seeded() {
        let s = ::test::rng().gen_iter::<u32>().take(256).collect::<Vec<u32>>();
        let mut ra: IsaacRng = SeedableRng::from_seed(s.as_slice());
        let mut rb: IsaacRng = SeedableRng::from_seed(s.as_slice());
        assert!(order::equals(ra.gen_ascii_chars().take(100),
                              rb.gen_ascii_chars().take(100)));
    }
    #[test]
    fn test_rng_64_rand_seeded() {
        let s = ::test::rng().gen_iter::<u64>().take(256).collect::<Vec<u64>>();
        let mut ra: Isaac64Rng = SeedableRng::from_seed(s.as_slice());
        let mut rb: Isaac64Rng = SeedableRng::from_seed(s.as_slice());
        assert!(order::equals(ra.gen_ascii_chars().take(100),
                              rb.gen_ascii_chars().take(100)));
    }

    #[test]
    fn test_rng_32_seeded() {
        let seed = &[1, 23, 456, 7890, 12345];
        let mut ra: IsaacRng = SeedableRng::from_seed(seed);
        let mut rb: IsaacRng = SeedableRng::from_seed(seed);
        assert!(order::equals(ra.gen_ascii_chars().take(100),
                              rb.gen_ascii_chars().take(100)));
    }
    #[test]
    fn test_rng_64_seeded() {
        let seed = &[1, 23, 456, 7890, 12345];
        let mut ra: Isaac64Rng = SeedableRng::from_seed(seed);
        let mut rb: Isaac64Rng = SeedableRng::from_seed(seed);
        assert!(order::equals(ra.gen_ascii_chars().take(100),
                              rb.gen_ascii_chars().take(100)));
    }

    #[test]
    fn test_rng_32_reseed() {
        let s = ::test::rng().gen_iter::<u32>().take(256).collect::<Vec<u32>>();
        let mut r: IsaacRng = SeedableRng::from_seed(s.as_slice());
        let string1: String = r.gen_ascii_chars().take(100).collect();

        r.reseed(s.as_slice());

        let string2: String = r.gen_ascii_chars().take(100).collect();
        assert_eq!(string1, string2);
    }
    #[test]
    fn test_rng_64_reseed() {
        let s = ::test::rng().gen_iter::<u64>().take(256).collect::<Vec<u64>>();
        let mut r: Isaac64Rng = SeedableRng::from_seed(s.as_slice());
        let string1: String = r.gen_ascii_chars().take(100).collect();

        r.reseed(s.as_slice());

        let string2: String = r.gen_ascii_chars().take(100).collect();
        assert_eq!(string1, string2);
    }

    #[test]
    fn test_rng_32_true_values() {
        let seed = &[1, 23, 456, 7890, 12345];
        let mut ra: IsaacRng = SeedableRng::from_seed(seed);
        // Regression test that isaac is actually using the above vector
        let v = Vec::from_fn(10, |_| ra.next_u32());
        assert_eq!(v,
                   vec!(2558573138, 873787463, 263499565, 2103644246, 3595684709,
                        4203127393, 264982119, 2765226902, 2737944514, 3900253796));

        let seed = &[12345, 67890, 54321, 9876];
        let mut rb: IsaacRng = SeedableRng::from_seed(seed);
        // skip forward to the 10000th number
        for _ in range(0u, 10000) { rb.next_u32(); }

        let v = Vec::from_fn(10, |_| rb.next_u32());
        assert_eq!(v,
                   vec!(3676831399, 3183332890, 2834741178, 3854698763, 2717568474,
                        1576568959, 3507990155, 179069555, 141456972, 2478885421));
    }
    #[test]
    fn test_rng_64_true_values() {
        let seed = &[1, 23, 456, 7890, 12345];
        let mut ra: Isaac64Rng = SeedableRng::from_seed(seed);
        // Regression test that isaac is actually using the above vector
        let v = Vec::from_fn(10, |_| ra.next_u64());
        assert_eq!(v,
                   vec!(547121783600835980, 14377643087320773276, 17351601304698403469,
                        1238879483818134882, 11952566807690396487, 13970131091560099343,
                        4469761996653280935, 15552757044682284409, 6860251611068737823,
                        13722198873481261842));

        let seed = &[12345, 67890, 54321, 9876];
        let mut rb: Isaac64Rng = SeedableRng::from_seed(seed);
        // skip forward to the 10000th number
        for _ in range(0u, 10000) { rb.next_u64(); }

        let v = Vec::from_fn(10, |_| rb.next_u64());
        assert_eq!(v,
                   vec!(18143823860592706164, 8491801882678285927, 2699425367717515619,
                        17196852593171130876, 2606123525235546165, 15790932315217671084,
                        596345674630742204, 9947027391921273664, 11788097613744130851,
                        10391409374914919106));
    }
}
