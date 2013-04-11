// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Random number generation

use int;
use prelude::*;
use str;
use task;
use u32;
use uint;
use util;
use vec;
use libc::size_t;

/// A type that can be randomly generated using an RNG
pub trait Rand {
    fn rand(rng: @rand::Rng) -> Self;
}

impl Rand for int {
    fn rand(rng: @rand::Rng) -> int {
        rng.gen_int()
    }
}

impl Rand for i8 {
    fn rand(rng: @rand::Rng) -> i8 {
        rng.gen_i8()
    }
}

impl Rand for i16 {
    fn rand(rng: @rand::Rng) -> i16 {
        rng.gen_i16()
    }
}

impl Rand for i32 {
    fn rand(rng: @rand::Rng) -> i32 {
        rng.gen_i32()
    }
}

impl Rand for i64 {
    fn rand(rng: @rand::Rng) -> i64 {
        rng.gen_i64()
    }
}

impl Rand for u8 {
    fn rand(rng: @rand::Rng) -> u8 {
        rng.gen_u8()
    }
}

impl Rand for u16 {
    fn rand(rng: @rand::Rng) -> u16 {
        rng.gen_u16()
    }
}

impl Rand for u32 {
    fn rand(rng: @rand::Rng) -> u32 {
        rng.gen_u32()
    }
}

impl Rand for u64 {
    fn rand(rng: @rand::Rng) -> u64 {
        rng.gen_u64()
    }
}

impl Rand for float {
    fn rand(rng: @rand::Rng) -> float {
        rng.gen_float()
    }
}

impl Rand for f32 {
    fn rand(rng: @rand::Rng) -> f32 {
        rng.gen_f32()
    }
}

impl Rand for f64 {
    fn rand(rng: @rand::Rng) -> f64 {
        rng.gen_f64()
    }
}

impl Rand for char {
    fn rand(rng: @rand::Rng) -> char {
        rng.gen_char()
    }
}

impl Rand for bool {
    fn rand(rng: @rand::Rng) -> bool {
        rng.gen_bool()
    }
}

impl<T:Rand> Rand for Option<T> {
    fn rand(rng: @rand::Rng) -> Option<T> {
        if rng.gen_bool() {
            Some(Rand::rand(rng))
        } else {
            None
        }
    }
}

#[allow(non_camel_case_types)] // runtime type
pub enum rust_rng {}

#[abi = "cdecl"]
pub mod rustrt {
    use libc::size_t;
    use rand::rust_rng;

    pub extern {
        unsafe fn rand_seed_size() -> size_t;
        unsafe fn rand_gen_seed(buf: *mut u8, sz: size_t);
        unsafe fn rand_new_seeded(buf: *u8, sz: size_t) -> *rust_rng;
        unsafe fn rand_next(rng: *rust_rng) -> u32;
        unsafe fn rand_free(rng: *rust_rng);
    }
}

/// A random number generator
pub trait Rng {
    /// Return the next random integer
    fn next(&self) -> u32;
}

/// A value with a particular weight compared to other values
pub struct Weighted<T> {
    weight: uint,
    item: T,
}

pub trait RngUtil {
    fn gen<T:Rand>(&self) -> T;
    /**
     * Return a random int
     *
     * *Example*
     *
     * ~~~
     *
     * use core::rand::RngUtil;
     *
     * fn main() {
     *     rng = rand::Rng();
     *     println(fmt!("%d",rng.gen_int()));
     * }
     * ~~~
     */
    fn gen_int(&self) -> int;
    fn gen_int_range(&self, start: int, end: int) -> int;
    /// Return a random i8
    fn gen_i8(&self) -> i8;
    /// Return a random i16
    fn gen_i16(&self) -> i16;
    /// Return a random i32
    fn gen_i32(&self) -> i32;
    /// Return a random i64
    fn gen_i64(&self) -> i64;
    /// Return a random uint
    fn gen_uint(&self) -> uint;
    /**
     * Return a uint randomly chosen from the range [start, end),
     * failing if start >= end
     */
    fn gen_uint_range(&self, start: uint, end: uint) -> uint;
    /// Return a random u8
    fn gen_u8(&self) -> u8;
    /// Return a random u16
    fn gen_u16(&self) -> u16;
    /// Return a random u32
    fn gen_u32(&self) -> u32;
    /// Return a random u64
    fn gen_u64(&self) -> u64;
    /**
     * Return random float in the interval [0,1]
     *
     * *Example*
     *
     * ~~~
     *
     * use core::rand::RngUtil;
     *
     * fn main() {
     *     rng = rand::Rng();
     *     println(fmt!("%f",rng.gen_float()));
     * }
     * ~~~
     */
    fn gen_float(&self) -> float;
    /// Return a random f32 in the interval [0,1]
    fn gen_f32(&self) -> f32;
    /// Return a random f64 in the interval [0,1]
    fn gen_f64(&self) -> f64;
    /// Return a random char
    fn gen_char(&self) -> char;
    /**
     * Return a char randomly chosen from chars, failing if chars is empty
     */
    fn gen_char_from(&self, chars: &str) -> char;
    /**
     * Return a random bool
     *
     * *Example*
     *
     * ~~~
     *
     * use core::rand::RngUtil;
     *
     * fn main() {
     *     rng = rand::Rng();
     *     println(fmt!("%b",rng.gen_bool()));
     * }
     * ~~~
     */
    fn gen_bool(&self) -> bool;
    /**
     * Return a bool with a 1 in n chance of true
     *
     * *Example*
     *
     * ~~~
     *
     * use core::rand::RngUtil;
     *
     * fn main() {
     *     rng = rand::Rng();
     *     println(fmt!("%b",rng.gen_weighted_bool(3)));
     * }
     * ~~~
     */
    fn gen_weighted_bool(&self, n: uint) -> bool;
    /**
     * Return a random string of the specified length composed of A-Z,a-z,0-9
     *
     * *Example*
     *
     * ~~~
     *
     * use core::rand::RngUtil;
     *
     * fn main() {
     *     rng = rand::Rng();
     *     println(rng.gen_str(8));
     * }
     * ~~~
     */
    fn gen_str(&self, len: uint) -> ~str;
    /**
     * Return a random byte string of the specified length
     *
     * *Example*
     *
     * ~~~
     *
     * use core::rand::RngUtil;
     *
     * fn main() {
     *     rng = rand::Rng();
     *     println(fmt!("%?",rng.gen_bytes(8)));
     * }
     * ~~~
     */
    fn gen_bytes(&self, len: uint) -> ~[u8];
    ///
    /**
     * Choose an item randomly, failing if values is empty
     *
     * *Example*
     *
     * ~~~
     *
     * use core::rand::RngUtil;
     *
     * fn main() {
     *     rng = rand::Rng();
     *     println(fmt!("%d",rng.choose([1,2,4,8,16,32])));
     * }
     * ~~~
     */
    fn choose<T:Copy>(&self, values: &[T]) -> T;
    /// Choose Some(item) randomly, returning None if values is empty
    fn choose_option<T:Copy>(&self, values: &[T]) -> Option<T>;
    /**
     * Choose an item respecting the relative weights, failing if the sum of
     * the weights is 0
     *
     * *Example*
     *
     * ~~~
     *
     * use core::rand::RngUtil;
     *
     * fn main() {
     *     rng = rand::Rng();
     *     let x = [rand::Weighted {weight: 4, item: 'a'},
     *              rand::Weighted {weight: 2, item: 'b'},
     *              rand::Weighted {weight: 2, item: 'c'}];
     *     println(fmt!("%c",rng.choose_weighted(x)));
     * }
     * ~~~
     */
    fn choose_weighted<T:Copy>(&self, v : &[Weighted<T>]) -> T;
    /**
     * Choose Some(item) respecting the relative weights, returning none if
     * the sum of the weights is 0
     *
     * *Example*
     *
     * ~~~
     *
     * use core::rand::RngUtil;
     *
     * fn main() {
     *     rng = rand::Rng();
     *     let x = [rand::Weighted {weight: 4, item: 'a'},
     *              rand::Weighted {weight: 2, item: 'b'},
     *              rand::Weighted {weight: 2, item: 'c'}];
     *     println(fmt!("%?",rng.choose_weighted_option(x)));
     * }
     * ~~~
     */
    fn choose_weighted_option<T:Copy>(&self, v: &[Weighted<T>]) -> Option<T>;
    /**
     * Return a vec containing copies of the items, in order, where
     * the weight of the item determines how many copies there are
     *
     * *Example*
     *
     * ~~~
     *
     * use core::rand::RngUtil;
     *
     * fn main() {
     *     rng = rand::Rng();
     *     let x = [rand::Weighted {weight: 4, item: 'a'},
     *              rand::Weighted {weight: 2, item: 'b'},
     *              rand::Weighted {weight: 2, item: 'c'}];
     *     println(fmt!("%?",rng.weighted_vec(x)));
     * }
     * ~~~
     */
    fn weighted_vec<T:Copy>(&self, v: &[Weighted<T>]) -> ~[T];
    /**
     * Shuffle a vec
     *
     * *Example*
     *
     * ~~~
     *
     * use core::rand::RngUtil;
     *
     * fn main() {
     *     rng = rand::Rng();
     *     println(fmt!("%?",rng.shuffle([1,2,3])));
     * }
     * ~~~
     */
    fn shuffle<T:Copy>(&self, values: &[T]) -> ~[T];
    /**
     * Shuffle a mutable vec in place
     *
     * *Example*
     *
     * ~~~
     *
     * use core::rand::RngUtil;
     *
     * fn main() {
     *     rng = rand::Rng();
     *     let mut y = [1,2,3];
     *     rng.shuffle_mut(y);
     *     println(fmt!("%?",y));
     *     rng.shuffle_mut(y);
     *     println(fmt!("%?",y));
     * }
     * ~~~
     */
    fn shuffle_mut<T>(&self, values: &mut [T]);
}

/// Extension methods for random number generators
impl RngUtil for @Rng {
    /// Return a random value for a Rand type
    fn gen<T:Rand>(&self) -> T {
        Rand::rand(*self)
    }

    /// Return a random int
    fn gen_int(&self) -> int {
        self.gen_i64() as int
    }

    /**
     * Return an int randomly chosen from the range [start, end),
     * failing if start >= end
     */
    fn gen_int_range(&self, start: int, end: int) -> int {
        assert!(start < end);
        start + int::abs(self.gen_int() % (end - start))
    }

    /// Return a random i8
    fn gen_i8(&self) -> i8 {
        self.next() as i8
    }

    /// Return a random i16
    fn gen_i16(&self) -> i16 {
        self.next() as i16
    }

    /// Return a random i32
    fn gen_i32(&self) -> i32 {
        self.next() as i32
    }

    /// Return a random i64
    fn gen_i64(&self) -> i64 {
        (self.next() as i64 << 32) | self.next() as i64
    }

    /// Return a random uint
    fn gen_uint(&self) -> uint {
        self.gen_u64() as uint
    }

    /**
     * Return a uint randomly chosen from the range [start, end),
     * failing if start >= end
     */
    fn gen_uint_range(&self, start: uint, end: uint) -> uint {
        assert!(start < end);
        start + (self.gen_uint() % (end - start))
    }

    /// Return a random u8
    fn gen_u8(&self) -> u8 {
        self.next() as u8
    }

    /// Return a random u16
    fn gen_u16(&self) -> u16 {
        self.next() as u16
    }

    /// Return a random u32
    fn gen_u32(&self) -> u32 {
        self.next()
    }

    /// Return a random u64
    fn gen_u64(&self) -> u64 {
        (self.next() as u64 << 32) | self.next() as u64
    }

    /// Return a random float in the interval [0,1]
    fn gen_float(&self) -> float {
        self.gen_f64() as float
    }

    /// Return a random f32 in the interval [0,1]
    fn gen_f32(&self) -> f32 {
        self.gen_f64() as f32
    }

    /// Return a random f64 in the interval [0,1]
    fn gen_f64(&self) -> f64 {
        let u1 = self.next() as f64;
        let u2 = self.next() as f64;
        let u3 = self.next() as f64;
        static scale : f64 = (u32::max_value as f64) + 1.0f64;
        return ((u1 / scale + u2) / scale + u3) / scale;
    }

    /// Return a random char
    fn gen_char(&self) -> char {
        self.next() as char
    }

    /**
     * Return a char randomly chosen from chars, failing if chars is empty
     */
    fn gen_char_from(&self, chars: &str) -> char {
        assert!(!chars.is_empty());
        let mut cs = ~[];
        for str::each_char(chars) |c| { cs.push(c) }
        self.choose(cs)
    }

    /// Return a random bool
    fn gen_bool(&self) -> bool {
        self.next() & 1u32 == 1u32
    }

    /// Return a bool with a 1-in-n chance of true
    fn gen_weighted_bool(&self, n: uint) -> bool {
        if n == 0u {
            true
        } else {
            self.gen_uint_range(1u, n + 1u) == 1u
        }
    }

    /**
     * Return a random string of the specified length composed of A-Z,a-z,0-9
     */
    fn gen_str(&self, len: uint) -> ~str {
        let charset = ~"ABCDEFGHIJKLMNOPQRSTUVWXYZ\
                       abcdefghijklmnopqrstuvwxyz\
                       0123456789";
        let mut s = ~"";
        let mut i = 0u;
        while (i < len) {
            s = s + str::from_char(self.gen_char_from(charset));
            i += 1u;
        }
        s
    }

    /// Return a random byte string of the specified length
    fn gen_bytes(&self, len: uint) -> ~[u8] {
        do vec::from_fn(len) |_i| {
            self.gen_u8()
        }
    }

    /// Choose an item randomly, failing if values is empty
    fn choose<T:Copy>(&self, values: &[T]) -> T {
        self.choose_option(values).get()
    }

    /// Choose Some(item) randomly, returning None if values is empty
    fn choose_option<T:Copy>(&self, values: &[T]) -> Option<T> {
        if values.is_empty() {
            None
        } else {
            Some(values[self.gen_uint_range(0u, values.len())])
        }
    }

    /**
     * Choose an item respecting the relative weights, failing if the sum of
     * the weights is 0
     */
    fn choose_weighted<T:Copy>(&self, v : &[Weighted<T>]) -> T {
        self.choose_weighted_option(v).get()
    }

    /**
     * Choose Some(item) respecting the relative weights, returning none if
     * the sum of the weights is 0
     */
    fn choose_weighted_option<T:Copy>(&self, v: &[Weighted<T>]) -> Option<T> {
        let mut total = 0u;
        for v.each |item| {
            total += item.weight;
        }
        if total == 0u {
            return None;
        }
        let chosen = self.gen_uint_range(0u, total);
        let mut so_far = 0u;
        for v.each |item| {
            so_far += item.weight;
            if so_far > chosen {
                return Some(item.item);
            }
        }
        util::unreachable();
    }

    /**
     * Return a vec containing copies of the items, in order, where
     * the weight of the item determines how many copies there are
     */
    fn weighted_vec<T:Copy>(&self, v: &[Weighted<T>]) -> ~[T] {
        let mut r = ~[];
        for v.each |item| {
            for uint::range(0u, item.weight) |_i| {
                r.push(item.item);
            }
        }
        r
    }

    /// Shuffle a vec
    fn shuffle<T:Copy>(&self, values: &[T]) -> ~[T] {
        let mut m = vec::from_slice(values);
        self.shuffle_mut(m);
        m
    }

    /// Shuffle a mutable vec in place
    fn shuffle_mut<T>(&self, values: &mut [T]) {
        let mut i = values.len();
        while i >= 2u {
            // invariant: elements with index >= i have been locked in place.
            i -= 1u;
            // lock element i in place.
            vec::swap(values, i, self.gen_uint_range(0u, i + 1u));
        }
    }

}

struct RandRes {
    rng: *rust_rng,
}

impl Drop for RandRes {
    fn finalize(&self) {
        unsafe {
            rustrt::rand_free(self.rng);
        }
    }
}

fn RandRes(rng: *rust_rng) -> RandRes {
    RandRes {
        rng: rng
    }
}

impl Rng for @RandRes {
    fn next(&self) -> u32 {
        unsafe {
            return rustrt::rand_next((*self).rng);
        }
    }
}

/// Create a new random seed for seeded_rng
pub fn seed() -> ~[u8] {
    unsafe {
        let n = rustrt::rand_seed_size() as uint;
        let mut s = vec::from_elem(n, 0_u8);
        do vec::as_mut_buf(s) |p, sz| {
            rustrt::rand_gen_seed(p, sz as size_t)
        }
        s
    }
}

/// Create a random number generator with a system specified seed
pub fn Rng() -> @Rng {
    seeded_rng(seed())
}

/**
 * Create a random number generator using the specified seed. A generator
 * constructed with a given seed will generate the same sequence of values as
 * all other generators constructed with the same seed. The seed may be any
 * length.
 */
pub fn seeded_rng(seed: &[u8]) -> @Rng {
    @seeded_randres(seed) as @Rng
}

fn seeded_randres(seed: &[u8]) -> @RandRes {
    unsafe {
        do vec::as_imm_buf(seed) |p, sz| {
            @RandRes(rustrt::rand_new_seeded(p, sz as size_t))
        }
    }
}

struct XorShiftState {
    mut x: u32,
    mut y: u32,
    mut z: u32,
    mut w: u32,
}

impl Rng for XorShiftState {
    fn next(&self) -> u32 {
        let x = self.x;
        let mut t = x ^ (x << 11);
        self.x = self.y;
        self.y = self.z;
        self.z = self.w;
        let w = self.w;
        self.w = w ^ (w >> 19) ^ (t ^ (t >> 8));
        self.w
    }
}

pub fn xorshift() -> @Rng {
    // constants taken from http://en.wikipedia.org/wiki/Xorshift
    seeded_xorshift(123456789u32, 362436069u32, 521288629u32, 88675123u32)
}

pub fn seeded_xorshift(x: u32, y: u32, z: u32, w: u32) -> @Rng {
    @XorShiftState { x: x, y: y, z: z, w: w } as @Rng
}


// used to make space in TLS for a random number generator
fn tls_rng_state(_v: @RandRes) {}

/**
 * Gives back a lazily initialized task-local random number generator,
 * seeded by the system. Intended to be used in method chaining style, ie
 * task_rng().gen_int().
 */
pub fn task_rng() -> @Rng {
    let r : Option<@RandRes>;
    unsafe {
        r = task::local_data::local_data_get(tls_rng_state);
    }
    match r {
        None => {
            unsafe {
                let rng = seeded_randres(seed());
                task::local_data::local_data_set(tls_rng_state, rng);
                @rng as @Rng
            }
        }
        Some(rng) => @rng as @Rng
    }
}

/**
 * Returns a random uint, using the task's based random number generator.
 */
pub fn random() -> uint {
    task_rng().gen_uint()
}


#[cfg(test)]
pub mod tests {
    use option::{Option, Some};
    use rand;

    #[test]
    pub fn rng_seeded() {
        let seed = rand::seed();
        let ra = rand::seeded_rng(seed);
        let rb = rand::seeded_rng(seed);
        assert!(ra.gen_str(100u) == rb.gen_str(100u));
    }

    #[test]
    pub fn rng_seeded_custom_seed() {
        // much shorter than generated seeds which are 1024 bytes
        let seed = [2u8, 32u8, 4u8, 32u8, 51u8];
        let ra = rand::seeded_rng(seed);
        let rb = rand::seeded_rng(seed);
        assert!(ra.gen_str(100u) == rb.gen_str(100u));
    }

    #[test]
    pub fn rng_seeded_custom_seed2() {
        let seed = [2u8, 32u8, 4u8, 32u8, 51u8];
        let ra = rand::seeded_rng(seed);
        // Regression test that isaac is actually using the above vector
        let r = ra.next();
        error!("%?", r);
        assert!(r == 890007737u32 // on x86_64
                     || r == 2935188040u32); // on x86
    }

    #[test]
    pub fn gen_int_range() {
        let r = rand::Rng();
        let a = r.gen_int_range(-3, 42);
        assert!(a >= -3 && a < 42);
        assert!(r.gen_int_range(0, 1) == 0);
        assert!(r.gen_int_range(-12, -11) == -12);
    }

    #[test]
    #[should_fail]
    #[ignore(cfg(windows))]
    pub fn gen_int_from_fail() {
        rand::Rng().gen_int_range(5, -2);
    }

    #[test]
    pub fn gen_uint_range() {
        let r = rand::Rng();
        let a = r.gen_uint_range(3u, 42u);
        assert!(a >= 3u && a < 42u);
        assert!(r.gen_uint_range(0u, 1u) == 0u);
        assert!(r.gen_uint_range(12u, 13u) == 12u);
    }

    #[test]
    #[should_fail]
    #[ignore(cfg(windows))]
    pub fn gen_uint_range_fail() {
        rand::Rng().gen_uint_range(5u, 2u);
    }

    #[test]
    pub fn gen_float() {
        let r = rand::Rng();
        let a = r.gen_float();
        let b = r.gen_float();
        debug!((a, b));
    }

    #[test]
    pub fn gen_weighted_bool() {
        let r = rand::Rng();
        assert!(r.gen_weighted_bool(0u) == true);
        assert!(r.gen_weighted_bool(1u) == true);
    }

    #[test]
    pub fn gen_str() {
        let r = rand::Rng();
        debug!(r.gen_str(10u));
        debug!(r.gen_str(10u));
        debug!(r.gen_str(10u));
        assert!(r.gen_str(0u).len() == 0u);
        assert!(r.gen_str(10u).len() == 10u);
        assert!(r.gen_str(16u).len() == 16u);
    }

    #[test]
    pub fn gen_bytes() {
        let r = rand::Rng();
        assert!(r.gen_bytes(0u).len() == 0u);
        assert!(r.gen_bytes(10u).len() == 10u);
        assert!(r.gen_bytes(16u).len() == 16u);
    }

    #[test]
    pub fn choose() {
        let r = rand::Rng();
        assert!(r.choose([1, 1, 1]) == 1);
    }

    #[test]
    pub fn choose_option() {
        let r = rand::Rng();
        let x: Option<int> = r.choose_option([]);
        assert!(x.is_none());
        assert!(r.choose_option([1, 1, 1]) == Some(1));
    }

    #[test]
    pub fn choose_weighted() {
        let r = rand::Rng();
        assert!(r.choose_weighted(~[
            rand::Weighted { weight: 1u, item: 42 },
        ]) == 42);
        assert!(r.choose_weighted(~[
            rand::Weighted { weight: 0u, item: 42 },
            rand::Weighted { weight: 1u, item: 43 },
        ]) == 43);
    }

    #[test]
    pub fn choose_weighted_option() {
        let r = rand::Rng();
        assert!(r.choose_weighted_option(~[
            rand::Weighted { weight: 1u, item: 42 },
        ]) == Some(42));
        assert!(r.choose_weighted_option(~[
            rand::Weighted { weight: 0u, item: 42 },
            rand::Weighted { weight: 1u, item: 43 },
        ]) == Some(43));
        let v: Option<int> = r.choose_weighted_option([]);
        assert!(v.is_none());
    }

    #[test]
    pub fn weighted_vec() {
        let r = rand::Rng();
        let empty: ~[int] = ~[];
        assert!(r.weighted_vec(~[]) == empty);
        assert!(r.weighted_vec(~[
            rand::Weighted { weight: 0u, item: 3u },
            rand::Weighted { weight: 1u, item: 2u },
            rand::Weighted { weight: 2u, item: 1u },
        ]) == ~[2u, 1u, 1u]);
    }

    #[test]
    pub fn shuffle() {
        let r = rand::Rng();
        let empty: ~[int] = ~[];
        assert!(r.shuffle(~[]) == empty);
        assert!(r.shuffle(~[1, 1, 1]) == ~[1, 1, 1]);
    }

    #[test]
    pub fn task_rng() {
        let r = rand::task_rng();
        r.gen_int();
        assert!(r.shuffle(~[1, 1, 1]) == ~[1, 1, 1]);
        assert!(r.gen_uint_range(0u, 1u) == 0u);
    }

    #[test]
    pub fn random() {
        // not sure how to test this aside from just getting a number
        let _n : uint = rand::random();
    }
}


// Local Variables:
// mode: rust;
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// End:
