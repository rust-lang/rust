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

// NB: transitional, de-mode-ing.
#[forbid(deprecated_mode)];
#[forbid(deprecated_pattern)];

use int;
use prelude::*;
use str;
use task;
use u32;
use uint;
use util;
use vec;

#[allow(non_camel_case_types)] // runtime type
enum rctx {}

#[abi = "cdecl"]
extern mod rustrt {
    unsafe fn rand_seed() -> ~[u8];
    unsafe fn rand_new() -> *rctx;
    unsafe fn rand_new_seeded2(&&seed: ~[u8]) -> *rctx;
    unsafe fn rand_next(c: *rctx) -> u32;
    unsafe fn rand_free(c: *rctx);
}

/// A random number generator
pub trait Rng {
    /// Return the next random integer
    fn next() -> u32;
}

/// A value with a particular weight compared to other values
pub struct Weighted<T> {
    weight: uint,
    item: T,
}

/// Extension methods for random number generators
impl Rng {

    /// Return a random int
    fn gen_int() -> int {
        self.gen_i64() as int
    }

    /**
     * Return an int randomly chosen from the range [start, end),
     * failing if start >= end
     */
    fn gen_int_range(start: int, end: int) -> int {
        assert start < end;
        start + int::abs(self.gen_int() % (end - start))
    }

    /// Return a random i8
    fn gen_i8() -> i8 {
        self.next() as i8
    }

    /// Return a random i16
    fn gen_i16() -> i16 {
        self.next() as i16
    }

    /// Return a random i32
    fn gen_i32() -> i32 {
        self.next() as i32
    }

    /// Return a random i64
    fn gen_i64() -> i64 {
        (self.next() as i64 << 32) | self.next() as i64
    }

    /// Return a random uint
    fn gen_uint() -> uint {
        self.gen_u64() as uint
    }

    /**
     * Return a uint randomly chosen from the range [start, end),
     * failing if start >= end
     */
    fn gen_uint_range(start: uint, end: uint) -> uint {
        assert start < end;
        start + (self.gen_uint() % (end - start))
    }

    /// Return a random u8
    fn gen_u8() -> u8 {
        self.next() as u8
    }

    /// Return a random u16
    fn gen_u16() -> u16 {
        self.next() as u16
    }

    /// Return a random u32
    fn gen_u32() -> u32 {
        self.next()
    }

    /// Return a random u64
    fn gen_u64() -> u64 {
        (self.next() as u64 << 32) | self.next() as u64
    }

    /// Return a random float in the interval [0,1]
    fn gen_float() -> float {
        self.gen_f64() as float
    }

    /// Return a random f32 in the interval [0,1]
    fn gen_f32() -> f32 {
        self.gen_f64() as f32
    }

    /// Return a random f64 in the interval [0,1]
    fn gen_f64() -> f64 {
        let u1 = self.next() as f64;
        let u2 = self.next() as f64;
        let u3 = self.next() as f64;
        const scale : f64 = (u32::max_value as f64) + 1.0f64;
        return ((u1 / scale + u2) / scale + u3) / scale;
    }

    /// Return a random char
    fn gen_char() -> char {
        self.next() as char
    }

    /**
     * Return a char randomly chosen from chars, failing if chars is empty
     */
    fn gen_char_from(chars: &str) -> char {
        assert !chars.is_empty();
        self.choose(str::chars(chars))
    }

    /// Return a random bool
    fn gen_bool() -> bool {
        self.next() & 1u32 == 1u32
    }

    /// Return a bool with a 1 in n chance of true
    fn gen_weighted_bool(n: uint) -> bool {
        if n == 0u {
            true
        } else {
            self.gen_uint_range(1u, n + 1u) == 1u
        }
    }

    /**
     * Return a random string of the specified length composed of A-Z,a-z,0-9
     */
    fn gen_str(len: uint) -> ~str {
        let charset = ~"ABCDEFGHIJKLMNOPQRSTUVWXYZ\
                       abcdefghijklmnopqrstuvwxyz\
                       0123456789";
        let mut s = ~"";
        let mut i = 0u;
        while (i < len) {
            s = s + str::from_char(self.gen_char_from(charset));
            i += 1u;
        }
        move s
    }

    /// Return a random byte string of the specified length
    fn gen_bytes(len: uint) -> ~[u8] {
        do vec::from_fn(len) |_i| {
            self.gen_u8()
        }
    }

    /// Choose an item randomly, failing if values is empty
    fn choose<T:Copy>(values: &[T]) -> T {
        self.choose_option(values).get()
    }

    /// Choose Some(item) randomly, returning None if values is empty
    fn choose_option<T:Copy>(values: &[T]) -> Option<T> {
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
    fn choose_weighted<T: Copy>(v : &[Weighted<T>]) -> T {
        self.choose_weighted_option(v).get()
    }

    /**
     * Choose Some(item) respecting the relative weights, returning none if
     * the sum of the weights is 0
     */
    fn choose_weighted_option<T:Copy>(v: &[Weighted<T>]) -> Option<T> {
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
    fn weighted_vec<T:Copy>(v: &[Weighted<T>]) -> ~[T] {
        let mut r = ~[];
        for v.each |item| {
            for uint::range(0u, item.weight) |_i| {
                r.push(item.item);
            }
        }
        move r
    }

    /// Shuffle a vec
    fn shuffle<T:Copy>(values: &[T]) -> ~[T] {
        let mut m = vec::from_slice(values);
        self.shuffle_mut(m);
        move m
    }

    /// Shuffle a mutable vec in place
    fn shuffle_mut<T>(values: &mut [T]) {
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
    c: *rctx,
    drop {
        unsafe {
            rustrt::rand_free(self.c);
        }
    }
}

fn RandRes(c: *rctx) -> RandRes {
    RandRes {
        c: c
    }
}

impl @RandRes: Rng {
    fn next() -> u32 {
        unsafe {
            return rustrt::rand_next((*self).c);
        }
    }
}

/// Create a new random seed for seeded_rng
pub fn seed() -> ~[u8] {
    unsafe {
        rustrt::rand_seed()
    }
}

/// Create a random number generator with a system specified seed
pub fn Rng() -> Rng {
    unsafe {
        @RandRes(rustrt::rand_new()) as Rng
    }
}

/**
 * Create a random number generator using the specified seed. A generator
 * constructed with a given seed will generate the same sequence of values as
 * all other generators constructed with the same seed. The seed may be any
 * length.
 */
pub fn seeded_rng(seed: &~[u8]) -> Rng {
    unsafe {
        @RandRes(rustrt::rand_new_seeded2(*seed)) as Rng
    }
}

struct XorShiftState {
    mut x: u32,
    mut y: u32,
    mut z: u32,
    mut w: u32,
}

impl XorShiftState: Rng {
    fn next() -> u32 {
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

pub pure fn xorshift() -> Rng {
    // constants taken from http://en.wikipedia.org/wiki/Xorshift
    seeded_xorshift(123456789u32, 362436069u32, 521288629u32, 88675123u32)
}

pub pure fn seeded_xorshift(x: u32, y: u32, z: u32, w: u32) -> Rng {
    XorShiftState { x: x, y: y, z: z, w: w } as Rng
}


// used to make space in TLS for a random number generator
fn tls_rng_state(_v: @RandRes) {}

/**
 * Gives back a lazily initialized task-local random number generator,
 * seeded by the system. Intended to be used in method chaining style, ie
 * task_rng().gen_int().
 */
pub fn task_rng() -> Rng {
    let r : Option<@RandRes>;
    unsafe {
        r = task::local_data::local_data_get(tls_rng_state);
    }
    match r {
        None => {
            unsafe {
                let rng = @RandRes(rustrt::rand_new());
                task::local_data::local_data_set(tls_rng_state, rng);
                rng as Rng
            }
        }
        Some(rng) => rng as Rng
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
    use debug;
    use option::{None, Option, Some};
    use rand;

    #[test]
    pub fn rng_seeded() {
        let seed = rand::seed();
        let ra = rand::seeded_rng(&seed);
        let rb = rand::seeded_rng(&seed);
        assert ra.gen_str(100u) == rb.gen_str(100u);
    }

    #[test]
    pub fn rng_seeded_custom_seed() {
        // much shorter than generated seeds which are 1024 bytes
        let seed = ~[2u8, 32u8, 4u8, 32u8, 51u8];
        let ra = rand::seeded_rng(&seed);
        let rb = rand::seeded_rng(&seed);
        assert ra.gen_str(100u) == rb.gen_str(100u);
    }

    #[test]
    pub fn rng_seeded_custom_seed2() {
        let seed = ~[2u8, 32u8, 4u8, 32u8, 51u8];
        let ra = rand::seeded_rng(&seed);
        // Regression test that isaac is actually using the above vector
        let r = ra.next();
        error!("%?", r);
        assert r == 890007737u32 // on x86_64
            || r == 2935188040u32; // on x86
    }

    #[test]
    pub fn gen_int_range() {
        let r = rand::Rng();
        let a = r.gen_int_range(-3, 42);
        assert a >= -3 && a < 42;
        assert r.gen_int_range(0, 1) == 0;
        assert r.gen_int_range(-12, -11) == -12;
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
        assert a >= 3u && a < 42u;
        assert r.gen_uint_range(0u, 1u) == 0u;
        assert r.gen_uint_range(12u, 13u) == 12u;
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
        log(debug, (a, b));
    }

    #[test]
    pub fn gen_weighted_bool() {
        let r = rand::Rng();
        assert r.gen_weighted_bool(0u) == true;
        assert r.gen_weighted_bool(1u) == true;
    }

    #[test]
    pub fn gen_str() {
        let r = rand::Rng();
        log(debug, r.gen_str(10u));
        log(debug, r.gen_str(10u));
        log(debug, r.gen_str(10u));
        assert r.gen_str(0u).len() == 0u;
        assert r.gen_str(10u).len() == 10u;
        assert r.gen_str(16u).len() == 16u;
    }

    #[test]
    pub fn gen_bytes() {
        let r = rand::Rng();
        assert r.gen_bytes(0u).len() == 0u;
        assert r.gen_bytes(10u).len() == 10u;
        assert r.gen_bytes(16u).len() == 16u;
    }

    #[test]
    pub fn choose() {
        let r = rand::Rng();
        assert r.choose([1, 1, 1]) == 1;
    }

    #[test]
    pub fn choose_option() {
        let r = rand::Rng();
        let x: Option<int> = r.choose_option([]);
        assert x.is_none();
        assert r.choose_option([1, 1, 1]) == Some(1);
    }

    #[test]
    pub fn choose_weighted() {
        let r = rand::Rng();
        assert r.choose_weighted(~[
            rand::Weighted { weight: 1u, item: 42 },
        ]) == 42;
        assert r.choose_weighted(~[
            rand::Weighted { weight: 0u, item: 42 },
            rand::Weighted { weight: 1u, item: 43 },
        ]) == 43;
    }

    #[test]
    pub fn choose_weighted_option() {
        let r = rand::Rng();
        assert r.choose_weighted_option(~[
            rand::Weighted { weight: 1u, item: 42 },
        ]) == Some(42);
        assert r.choose_weighted_option(~[
            rand::Weighted { weight: 0u, item: 42 },
            rand::Weighted { weight: 1u, item: 43 },
        ]) == Some(43);
        let v: Option<int> = r.choose_weighted_option([]);
        assert v.is_none();
    }

    #[test]
    pub fn weighted_vec() {
        let r = rand::Rng();
        let empty: ~[int] = ~[];
        assert r.weighted_vec(~[]) == empty;
        assert r.weighted_vec(~[
            rand::Weighted { weight: 0u, item: 3u },
            rand::Weighted { weight: 1u, item: 2u },
            rand::Weighted { weight: 2u, item: 1u },
        ]) == ~[2u, 1u, 1u];
    }

    #[test]
    pub fn shuffle() {
        let r = rand::Rng();
        let empty: ~[int] = ~[];
        assert r.shuffle(~[]) == empty;
        assert r.shuffle(~[1, 1, 1]) == ~[1, 1, 1];
    }

    #[test]
    pub fn task_rng() {
        let r = rand::task_rng();
        r.gen_int();
        assert r.shuffle(~[1, 1, 1]) == ~[1, 1, 1];
        assert r.gen_uint_range(0u, 1u) == 0u;
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
