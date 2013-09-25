// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/*!
Random number generation.

The key functions are `random()` and `Rng::gen()`. These are polymorphic
and so can be used to generate any type that implements `Rand`. Type inference
means that often a simple call to `rand::random()` or `rng.gen()` will
suffice, but sometimes an annotation is required, e.g. `rand::random::<float>()`.

See the `distributions` submodule for sampling random numbers from
distributions like normal and exponential.

# Examples

```rust
use std::rand;
use std::rand::Rng;

fn main() {
    let mut rng = rand::rng();
    if rng.gen() { // bool
        printfln!("int: %d, uint: %u", rng.gen(), rng.gen())
    }
}
 ```

```rust
use std::rand;

fn main () {
    let tuple_ptr = rand::random::<~(f64, char)>();
    printfln!(tuple_ptr)
}
 ```
*/

use cast;
use cmp;
use container::Container;
use int;
use iter::{Iterator, range, range_step};
use local_data;
use prelude::*;
use str;
use sys;
use u32;
use u64;
use uint;
use vec;
use libc::size_t;

pub mod distributions;

/// A type that can be randomly generated using an Rng
pub trait Rand {
    /// Generates a random instance of this type using the specified source of
    /// randomness
    fn rand<R: Rng>(rng: &mut R) -> Self;
}

impl Rand for int {
    #[inline]
    fn rand<R: Rng>(rng: &mut R) -> int {
        if int::bits == 32 {
            rng.next() as int
        } else {
            rng.gen::<i64>() as int
        }
    }
}

impl Rand for i8 {
    #[inline]
    fn rand<R: Rng>(rng: &mut R) -> i8 {
        rng.next() as i8
    }
}

impl Rand for i16 {
    #[inline]
    fn rand<R: Rng>(rng: &mut R) -> i16 {
        rng.next() as i16
    }
}

impl Rand for i32 {
    #[inline]
    fn rand<R: Rng>(rng: &mut R) -> i32 {
        rng.next() as i32
    }
}

impl Rand for i64 {
    #[inline]
    fn rand<R: Rng>(rng: &mut R) -> i64 {
        (rng.next() as i64 << 32) | rng.next() as i64
    }
}

impl Rand for uint {
    #[inline]
    fn rand<R: Rng>(rng: &mut R) -> uint {
        if uint::bits == 32 {
            rng.next() as uint
        } else {
            rng.gen::<u64>() as uint
        }
    }
}

impl Rand for u8 {
    #[inline]
    fn rand<R: Rng>(rng: &mut R) -> u8 {
        rng.next() as u8
    }
}

impl Rand for u16 {
    #[inline]
    fn rand<R: Rng>(rng: &mut R) -> u16 {
        rng.next() as u16
    }
}

impl Rand for u32 {
    #[inline]
    fn rand<R: Rng>(rng: &mut R) -> u32 {
        rng.next()
    }
}

impl Rand for u64 {
    #[inline]
    fn rand<R: Rng>(rng: &mut R) -> u64 {
        (rng.next() as u64 << 32) | rng.next() as u64
    }
}

impl Rand for float {
    #[inline]
    fn rand<R: Rng>(rng: &mut R) -> float {
        rng.gen::<f64>() as float
    }
}

impl Rand for f32 {
    #[inline]
    fn rand<R: Rng>(rng: &mut R) -> f32 {
        rng.gen::<f64>() as f32
    }
}

static SCALE : f64 = (u32::max_value as f64) + 1.0f64;
impl Rand for f64 {
    #[inline]
    fn rand<R: Rng>(rng: &mut R) -> f64 {
        let u1 = rng.next() as f64;
        let u2 = rng.next() as f64;
        let u3 = rng.next() as f64;

        ((u1 / SCALE + u2) / SCALE + u3) / SCALE
    }
}

impl Rand for bool {
    #[inline]
    fn rand<R: Rng>(rng: &mut R) -> bool {
        rng.next() & 1u32 == 1u32
    }
}

macro_rules! tuple_impl {
    // use variables to indicate the arity of the tuple
    ($($tyvar:ident),* ) => {
        // the trailing commas are for the 1 tuple
        impl<
            $( $tyvar : Rand ),*
            > Rand for ( $( $tyvar ),* , ) {

            #[inline]
            fn rand<R: Rng>(_rng: &mut R) -> ( $( $tyvar ),* , ) {
                (
                    // use the $tyvar's to get the appropriate number of
                    // repeats (they're not actually needed)
                    $(
                        _rng.gen::<$tyvar>()
                    ),*
                    ,
                )
            }
        }
    }
}

impl Rand for () {
    #[inline]
    fn rand<R: Rng>(_: &mut R) -> () { () }
}
tuple_impl!{A}
tuple_impl!{A, B}
tuple_impl!{A, B, C}
tuple_impl!{A, B, C, D}
tuple_impl!{A, B, C, D, E}
tuple_impl!{A, B, C, D, E, F}
tuple_impl!{A, B, C, D, E, F, G}
tuple_impl!{A, B, C, D, E, F, G, H}
tuple_impl!{A, B, C, D, E, F, G, H, I}
tuple_impl!{A, B, C, D, E, F, G, H, I, J}

impl<T:Rand> Rand for Option<T> {
    #[inline]
    fn rand<R: Rng>(rng: &mut R) -> Option<T> {
        if rng.gen() {
            Some(rng.gen())
        } else {
            None
        }
    }
}

impl<T: Rand> Rand for ~T {
    #[inline]
    fn rand<R: Rng>(rng: &mut R) -> ~T { ~rng.gen() }
}

impl<T: Rand + 'static> Rand for @T {
    #[inline]
    fn rand<R: Rng>(rng: &mut R) -> @T { @rng.gen() }
}

#[abi = "cdecl"]
pub mod rustrt {
    use libc::size_t;

    extern {
        pub fn rand_gen_seed(buf: *mut u8, sz: size_t);
    }
}

/// A value with a particular weight compared to other values
pub struct Weighted<T> {
    /// The numerical weight of this item
    weight: uint,
    /// The actual item which is being weighted
    item: T,
}

/// A random number generator
pub trait Rng {
    /// Return the next random integer
    fn next(&mut self) -> u32;


    /// Return a random value of a Rand type.
    ///
    /// # Example
    ///
    /// ```rust
    /// use std::rand;
    ///
    /// fn main() {
    ///    let rng = rand::task_rng();
    ///    let x: uint = rng.gen();
    ///    printfln!(x);
    ///    printfln!(rng.gen::<(float, bool)>());
    /// }
    /// ```
    #[inline(always)]
    fn gen<T: Rand>(&mut self) -> T {
        Rand::rand(self)
    }

    /// Return a random vector of the specified length.
    ///
    /// # Example
    ///
    /// ```rust
    /// use std::rand;
    ///
    /// fn main() {
    ///    let rng = rand::task_rng();
    ///    let x: ~[uint] = rng.gen_vec(10);
    ///    printfln!(x);
    ///    printfln!(rng.gen_vec::<(float, bool)>(5));
    /// }
    /// ```
    fn gen_vec<T: Rand>(&mut self, len: uint) -> ~[T] {
        vec::from_fn(len, |_| self.gen())
    }

    /// Generate a random primitive integer in the range [`low`,
    /// `high`). Fails if `low >= high`.
    ///
    /// This gives a uniform distribution (assuming this RNG is itself
    /// uniform), even for edge cases like `gen_integer_range(0u8,
    /// 170)`, which a naive modulo operation would return numbers
    /// less than 85 with double the probability to those greater than
    /// 85.
    ///
    /// # Example
    ///
    /// ```rust
    /// use std::rand;
    ///
    /// fn main() {
    ///    let rng = rand::task_rng();
    ///    let n: uint = rng.gen_integer_range(0u, 10);
    ///    printfln!(n);
    ///    let m: i16 = rng.gen_integer_range(-40, 400);
    ///    printfln!(m);
    /// }
    /// ```
    fn gen_integer_range<T: Rand + Int>(&mut self, low: T, high: T) -> T {
        assert!(low < high, "RNG.gen_integer_range called with low >= high");
        let range = (high - low).to_u64();
        let accept_zone = u64::max_value - u64::max_value % range;
        loop {
            let rand = self.gen::<u64>();
            if rand < accept_zone {
                return low + NumCast::from(rand % range);
            }
        }
    }

    /// Return a bool with a 1 in n chance of true
    ///
    /// # Example
    ///
    /// ```rust
    /// use std::rand;
    /// use std::rand::Rng;
    ///
    /// fn main() {
    ///     let mut rng = rand::rng();
    ///     printfln!("%b", rng.gen_weighted_bool(3));
    /// }
    /// ```
    fn gen_weighted_bool(&mut self, n: uint) -> bool {
        n == 0 || self.gen_integer_range(0, n) == 0
    }

    /// Return a random string of the specified length composed of
    /// A-Z,a-z,0-9.
    ///
    /// # Example
    ///
    /// ```rust
    /// use std::rand;
    ///
    /// fn main() {
    ///    println(rand::task_rng().gen_ascii_str(10));
    /// }
    /// ```
    fn gen_ascii_str(&mut self, len: uint) -> ~str {
        static GEN_ASCII_STR_CHARSET: &'static [u8] = bytes!("ABCDEFGHIJKLMNOPQRSTUVWXYZ\
                                                             abcdefghijklmnopqrstuvwxyz\
                                                             0123456789");
        let mut s = str::with_capacity(len);
        for _ in range(0, len) {
            s.push_char(self.choose(GEN_ASCII_STR_CHARSET) as char)
        }
        s
    }

    /// Choose an item randomly, failing if `values` is empty.
    fn choose<T: Clone>(&mut self, values: &[T]) -> T {
        self.choose_option(values).expect("Rng.choose: `values` is empty").clone()
    }

    /// Choose `Some(&item)` randomly, returning `None` if values is
    /// empty.
    ///
    /// # Example
    ///
    /// ```rust
    /// use std::rand;
    ///
    /// fn main() {
    ///     printfln!(rand::task_rng().choose_option([1,2,4,8,16,32]));
    ///     printfln!(rand::task_rng().choose_option([]));
    /// }
    /// ```
    fn choose_option<'a, T>(&mut self, values: &'a [T]) -> Option<&'a T> {
        if values.is_empty() {
            None
        } else {
            Some(&values[self.gen_integer_range(0u, values.len())])
        }
    }

    /// Choose an item respecting the relative weights, failing if the sum of
    /// the weights is 0
    ///
    /// # Example
    ///
    /// ```rust
    /// use std::rand;
    /// use std::rand::Rng;
    ///
    /// fn main() {
    ///     let mut rng = rand::rng();
    ///     let x = [rand::Weighted {weight: 4, item: 'a'},
    ///              rand::Weighted {weight: 2, item: 'b'},
    ///              rand::Weighted {weight: 2, item: 'c'}];
    ///     printfln!("%c", rng.choose_weighted(x));
    /// }
    /// ```
    fn choose_weighted<T:Clone>(&mut self, v: &[Weighted<T>]) -> T {
        self.choose_weighted_option(v).expect("Rng.choose_weighted: total weight is 0")
    }

    /// Choose Some(item) respecting the relative weights, returning none if
    /// the sum of the weights is 0
    ///
    /// # Example
    ///
    /// ```rust
    /// use std::rand;
    /// use std::rand::Rng;
    ///
    /// fn main() {
    ///     let mut rng = rand::rng();
    ///     let x = [rand::Weighted {weight: 4, item: 'a'},
    ///              rand::Weighted {weight: 2, item: 'b'},
    ///              rand::Weighted {weight: 2, item: 'c'}];
    ///     printfln!(rng.choose_weighted_option(x));
    /// }
    /// ```
    fn choose_weighted_option<T:Clone>(&mut self, v: &[Weighted<T>])
                                       -> Option<T> {
        let mut total = 0u;
        for item in v.iter() {
            total += item.weight;
        }
        if total == 0u {
            return None;
        }
        let chosen = self.gen_integer_range(0u, total);
        let mut so_far = 0u;
        for item in v.iter() {
            so_far += item.weight;
            if so_far > chosen {
                return Some(item.item.clone());
            }
        }
        unreachable!();
    }

    /// Return a vec containing copies of the items, in order, where
    /// the weight of the item determines how many copies there are
    ///
    /// # Example
    ///
    /// ```rust
    /// use std::rand;
    /// use std::rand::Rng;
    ///
    /// fn main() {
    ///     let mut rng = rand::rng();
    ///     let x = [rand::Weighted {weight: 4, item: 'a'},
    ///              rand::Weighted {weight: 2, item: 'b'},
    ///              rand::Weighted {weight: 2, item: 'c'}];
    ///     printfln!(rng.weighted_vec(x));
    /// }
    /// ```
    fn weighted_vec<T:Clone>(&mut self, v: &[Weighted<T>]) -> ~[T] {
        let mut r = ~[];
        for item in v.iter() {
            for _ in range(0u, item.weight) {
                r.push(item.item.clone());
            }
        }
        r
    }

    /// Shuffle a vec
    ///
    /// # Example
    ///
    /// ```rust
    /// use std::rand;
    ///
    /// fn main() {
    ///     printfln!(rand::task_rng().shuffle(~[1,2,3]));
    /// }
    /// ```
    fn shuffle<T>(&mut self, values: ~[T]) -> ~[T] {
        let mut v = values;
        self.shuffle_mut(v);
        v
    }

    /// Shuffle a mutable vector in place.
    ///
    /// # Example
    ///
    /// ```rust
    /// use std::rand;
    ///
    /// fn main() {
    ///    let rng = rand::task_rng();
    ///    let mut y = [1,2,3];
    ///    rng.shuffle_mut(y);
    ///    printfln!(y);
    ///    rng.shuffle_mut(y);
    ///    printfln!(y);
    /// }
    /// ```
    fn shuffle_mut<T>(&mut self, values: &mut [T]) {
        let mut i = values.len();
        while i >= 2u {
            // invariant: elements with index >= i have been locked in place.
            i -= 1u;
            // lock element i in place.
            values.swap(i, self.gen_integer_range(0u, i + 1u));
        }
    }

    /// Randomly sample up to `n` elements from an iterator.
    ///
    /// # Example
    ///
    /// ```rust
    /// use std::rand;
    ///
    /// fn main() {
    ///    let rng = rand::task_rng();
    ///    let sample = rng.sample(range(1, 100), 5);
    ///    printfln!(sample);
    /// }
    /// ```
    fn sample<A, T: Iterator<A>>(&mut self, iter: T, n: uint) -> ~[A] {
        let mut reservoir : ~[A] = vec::with_capacity(n);
        for (i, elem) in iter.enumerate() {
            if i < n {
                reservoir.push(elem);
                loop
            }

            let k = self.gen_integer_range(0, i + 1);
            if k < reservoir.len() {
                reservoir[k] = elem
            }
        }
        reservoir
    }
}

/// Create a random number generator with a default algorithm and seed.
///
/// It returns the cryptographically-safest `Rng` algorithm currently
/// available in Rust. If you require a specifically seeded `Rng` for
/// consistency over time you should pick one algorithm and create the
/// `Rng` yourself.
pub fn rng() -> IsaacRng {
    IsaacRng::new()
}

/// Create a weak random number generator with a default algorithm and seed.
///
/// It returns the fastest `Rng` algorithm currently available in Rust without
/// consideration for cryptography or security. If you require a specifically
/// seeded `Rng` for consistency over time you should pick one algorithm and
/// create the `Rng` yourself.
pub fn weak_rng() -> XorShiftRng {
    XorShiftRng::new()
}

static RAND_SIZE_LEN: u32 = 8;
static RAND_SIZE: u32 = 1 << RAND_SIZE_LEN;

/// A random number generator that uses the [ISAAC
/// algorithm](http://en.wikipedia.org/wiki/ISAAC_%28cipher%29).
///
/// The ISAAC algorithm is suitable for cryptographic purposes.
pub struct IsaacRng {
    priv cnt: u32,
    priv rsl: [u32, .. RAND_SIZE],
    priv mem: [u32, .. RAND_SIZE],
    priv a: u32,
    priv b: u32,
    priv c: u32
}

impl IsaacRng {
    /// Create an ISAAC random number generator with a random seed.
    pub fn new() -> IsaacRng {
        IsaacRng::new_seeded(seed())
    }

    /// Create an ISAAC random number generator with a seed. This can be any
    /// length, although the maximum number of bytes used is 1024 and any more
    /// will be silently ignored. A generator constructed with a given seed
    /// will generate the same sequence of values as all other generators
    /// constructed with the same seed.
    pub fn new_seeded(seed: &[u8]) -> IsaacRng {
        let mut rng = IsaacRng {
            cnt: 0,
            rsl: [0, .. RAND_SIZE],
            mem: [0, .. RAND_SIZE],
            a: 0, b: 0, c: 0
        };

        let array_size = sys::size_of_val(&rng.rsl);
        let copy_length = cmp::min(array_size, seed.len());

        // manually create a &mut [u8] slice of randrsl to copy into.
        let dest = unsafe { cast::transmute((&mut rng.rsl, array_size)) };
        vec::bytes::copy_memory(dest, seed, copy_length);
        rng.init(true);
        rng
    }

    /// Create an ISAAC random number generator using the default
    /// fixed seed.
    pub fn new_unseeded() -> IsaacRng {
        let mut rng = IsaacRng {
            cnt: 0,
            rsl: [0, .. RAND_SIZE],
            mem: [0, .. RAND_SIZE],
            a: 0, b: 0, c: 0
        };
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

        do 4.times { mix!(); }

        if use_rsl {
            macro_rules! memloop (
                ($arr:expr) => {{
                    for i in range_step(0u32, RAND_SIZE, 8) {
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
            for i in range_step(0u32, RAND_SIZE, 8) {
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
    fn isaac(&mut self) {
        self.c += 1;
        // abbreviations
        let mut a = self.a;
        let mut b = self.b + self.c;

        static MIDPOINT: uint = RAND_SIZE as uint / 2;

        macro_rules! ind (($x:expr) => {
            self.mem[($x >> 2) & (RAND_SIZE - 1)]
        });
        macro_rules! rngstep(
            ($j:expr, $shift:expr) => {{
                let base = $j;
                let mix = if $shift < 0 {
                    a >> -$shift as uint
                } else {
                    a << $shift as uint
                };

                let x = self.mem[base  + mr_offset];
                a = (a ^ mix) + self.mem[base + m2_offset];
                let y = ind!(x) + a + b;
                self.mem[base + mr_offset] = y;

                b = ind!(y >> RAND_SIZE_LEN) + x;
                self.rsl[base + mr_offset] = b;
            }}
        );

        let r = [(0, MIDPOINT), (MIDPOINT, 0)];
        for &(mr_offset, m2_offset) in r.iter() {
            for i in range_step(0u, MIDPOINT, 4) {
                rngstep!(i + 0, 13);
                rngstep!(i + 1, -6);
                rngstep!(i + 2, 2);
                rngstep!(i + 3, -16);
            }
        }

        self.a = a;
        self.b = b;
        self.cnt = RAND_SIZE;
    }
}

impl Rng for IsaacRng {
    #[inline]
    fn next(&mut self) -> u32 {
        if self.cnt == 0 {
            // make some more numbers
            self.isaac();
        }
        self.cnt -= 1;
        self.rsl[self.cnt]
    }
}

/// An [Xorshift random number
/// generator](http://en.wikipedia.org/wiki/Xorshift).
///
/// The Xorshift algorithm is not suitable for cryptographic purposes
/// but is very fast. If you do not know for sure that it fits your
/// requirements, use a more secure one such as `IsaacRng`.
pub struct XorShiftRng {
    priv x: u32,
    priv y: u32,
    priv z: u32,
    priv w: u32,
}

impl Rng for XorShiftRng {
    #[inline]
    fn next(&mut self) -> u32 {
        let x = self.x;
        let t = x ^ (x << 11);
        self.x = self.y;
        self.y = self.z;
        self.z = self.w;
        let w = self.w;
        self.w = w ^ (w >> 19) ^ (t ^ (t >> 8));
        self.w
    }
}

impl XorShiftRng {
    /// Create an xor shift random number generator with a random seed.
    pub fn new() -> XorShiftRng {
        #[fixed_stack_segment]; #[inline(never)];

        // generate seeds the same way as seed(), except we have a spceific size
        let mut s = [0u8, ..16];
        loop {
            do s.as_mut_buf |p, sz| {
                unsafe {
                    rustrt::rand_gen_seed(p, sz as size_t);
                }
            }
            if !s.iter().all(|x| *x == 0) {
                break;
            }
        }
        let s: &[u32, ..4] = unsafe { cast::transmute(&s) };
        XorShiftRng::new_seeded(s[0], s[1], s[2], s[3])
    }

    /**
     * Create a random number generator using the specified seed. A generator
     * constructed with a given seed will generate the same sequence of values
     * as all other generators constructed with the same seed.
     */
    pub fn new_seeded(x: u32, y: u32, z: u32, w: u32) -> XorShiftRng {
        XorShiftRng {
            x: x,
            y: y,
            z: z,
            w: w,
        }
    }
}

/// Create a new random seed.
pub fn seed() -> ~[u8] {
    #[fixed_stack_segment]; #[inline(never)];

    unsafe {
        let n = RAND_SIZE * 4;
        let mut s = vec::from_elem(n as uint, 0_u8);
        do s.as_mut_buf |p, sz| {
            rustrt::rand_gen_seed(p, sz as size_t)
        }
        s
    }
}

// used to make space in TLS for a random number generator
local_data_key!(tls_rng_state: @@mut IsaacRng)

/**
 * Gives back a lazily initialized task-local random number generator,
 * seeded by the system. Intended to be used in method chaining style, ie
 * `task_rng().gen::<int>()`.
 */
#[inline]
pub fn task_rng() -> @mut IsaacRng {
    let r = local_data::get(tls_rng_state, |k| k.map(|&k| *k));
    match r {
        None => {
            let rng = @@mut IsaacRng::new_seeded(seed());
            local_data::set(tls_rng_state, rng);
            *rng
        }
        Some(rng) => *rng
    }
}

// Allow direct chaining with `task_rng`
impl<R: Rng> Rng for @mut R {
    #[inline]
    fn next(&mut self) -> u32 {
        (**self).next()
    }
}

/**
 * Returns a random value of a Rand type, using the task's random number
 * generator.
 */
#[inline]
pub fn random<T: Rand>() -> T {
    task_rng().gen()
}

#[cfg(test)]
mod test {
    use iter::{Iterator, range};
    use option::{Option, Some};
    use super::*;

    #[test]
    fn test_rng_seeded() {
        let seed = seed();
        let mut ra = IsaacRng::new_seeded(seed);
        let mut rb = IsaacRng::new_seeded(seed);
        assert_eq!(ra.gen_ascii_str(100u), rb.gen_ascii_str(100u));
    }

    #[test]
    fn test_rng_seeded_custom_seed() {
        // much shorter than generated seeds which are 1024 bytes
        let seed = [2u8, 32u8, 4u8, 32u8, 51u8];
        let mut ra = IsaacRng::new_seeded(seed);
        let mut rb = IsaacRng::new_seeded(seed);
        assert_eq!(ra.gen_ascii_str(100u), rb.gen_ascii_str(100u));
    }

    #[test]
    fn test_rng_seeded_custom_seed2() {
        let seed = [2u8, 32u8, 4u8, 32u8, 51u8];
        let mut ra = IsaacRng::new_seeded(seed);
        // Regression test that isaac is actually using the above vector
        let r = ra.next();
        error!("%?", r);
        assert!(r == 890007737u32 // on x86_64
                     || r == 2935188040u32); // on x86
    }

    #[test]
    fn test_gen_integer_range() {
        let mut r = rng();
        for _ in range(0, 1000) {
            let a = r.gen_integer_range(-3i, 42);
            assert!(a >= -3 && a < 42);
            assert_eq!(r.gen_integer_range(0, 1), 0);
            assert_eq!(r.gen_integer_range(-12, -11), -12);
        }

        for _ in range(0, 1000) {
            let a = r.gen_integer_range(10, 42);
            assert!(a >= 10 && a < 42);
            assert_eq!(r.gen_integer_range(0, 1), 0);
            assert_eq!(r.gen_integer_range(3_000_000u, 3_000_001), 3_000_000);
        }

    }

    #[test]
    #[should_fail]
    fn test_gen_integer_range_fail_int() {
        let mut r = rng();
        r.gen_integer_range(5i, -2);
    }

    #[test]
    #[should_fail]
    fn test_gen_integer_range_fail_uint() {
        let mut r = rng();
        r.gen_integer_range(5u, 2u);
    }

    #[test]
    fn test_gen_float() {
        let mut r = rng();
        let a = r.gen::<float>();
        let b = r.gen::<float>();
        debug!((a, b));
    }

    #[test]
    fn test_gen_weighted_bool() {
        let mut r = rng();
        assert_eq!(r.gen_weighted_bool(0u), true);
        assert_eq!(r.gen_weighted_bool(1u), true);
    }

    #[test]
    fn test_gen_ascii_str() {
        let mut r = rng();
        debug!(r.gen_ascii_str(10u));
        debug!(r.gen_ascii_str(10u));
        debug!(r.gen_ascii_str(10u));
        assert_eq!(r.gen_ascii_str(0u).len(), 0u);
        assert_eq!(r.gen_ascii_str(10u).len(), 10u);
        assert_eq!(r.gen_ascii_str(16u).len(), 16u);
    }

    #[test]
    fn test_gen_vec() {
        let mut r = rng();
        assert_eq!(r.gen_vec::<u8>(0u).len(), 0u);
        assert_eq!(r.gen_vec::<u8>(10u).len(), 10u);
        assert_eq!(r.gen_vec::<f64>(16u).len(), 16u);
    }

    #[test]
    fn test_choose() {
        let mut r = rng();
        assert_eq!(r.choose([1, 1, 1]), 1);
    }

    #[test]
    fn test_choose_option() {
        let mut r = rng();
        let v: &[int] = &[];
        assert!(r.choose_option(v).is_none());

        let i = 1;
        let v = [1,1,1];
        assert_eq!(r.choose_option(v), Some(&i));
    }

    #[test]
    fn test_choose_weighted() {
        let mut r = rng();
        assert!(r.choose_weighted([
            Weighted { weight: 1u, item: 42 },
        ]) == 42);
        assert!(r.choose_weighted([
            Weighted { weight: 0u, item: 42 },
            Weighted { weight: 1u, item: 43 },
        ]) == 43);
    }

    #[test]
    fn test_choose_weighted_option() {
        let mut r = rng();
        assert!(r.choose_weighted_option([
            Weighted { weight: 1u, item: 42 },
        ]) == Some(42));
        assert!(r.choose_weighted_option([
            Weighted { weight: 0u, item: 42 },
            Weighted { weight: 1u, item: 43 },
        ]) == Some(43));
        let v: Option<int> = r.choose_weighted_option([]);
        assert!(v.is_none());
    }

    #[test]
    fn test_weighted_vec() {
        let mut r = rng();
        let empty: ~[int] = ~[];
        assert_eq!(r.weighted_vec([]), empty);
        assert!(r.weighted_vec([
            Weighted { weight: 0u, item: 3u },
            Weighted { weight: 1u, item: 2u },
            Weighted { weight: 2u, item: 1u },
        ]) == ~[2u, 1u, 1u]);
    }

    #[test]
    fn test_shuffle() {
        let mut r = rng();
        let empty: ~[int] = ~[];
        assert_eq!(r.shuffle(~[]), empty);
        assert_eq!(r.shuffle(~[1, 1, 1]), ~[1, 1, 1]);
    }

    #[test]
    fn test_task_rng() {
        let mut r = task_rng();
        r.gen::<int>();
        assert_eq!(r.shuffle(~[1, 1, 1]), ~[1, 1, 1]);
        assert_eq!(r.gen_integer_range(0u, 1u), 0u);
    }

    #[test]
    fn test_random() {
        // not sure how to test this aside from just getting some values
        let _n : uint = random();
        let _f : f32 = random();
        let _o : Option<Option<i8>> = random();
        let _many : ((),
                     (~uint, @int, ~Option<~(@u32, ~(@bool,))>),
                     (u8, i8, u16, i16, u32, i32, u64, i64),
                     (f32, (f64, (float,)))) = random();
    }

    #[test]
    fn test_sample() {
        let MIN_VAL = 1;
        let MAX_VAL = 100;

        let mut r = rng();
        let vals = range(MIN_VAL, MAX_VAL).to_owned_vec();
        let small_sample = r.sample(vals.iter(), 5);
        let large_sample = r.sample(vals.iter(), vals.len() + 5);

        assert_eq!(small_sample.len(), 5);
        assert_eq!(large_sample.len(), vals.len());

        assert!(small_sample.iter().all(|e| {
            **e >= MIN_VAL && **e <= MAX_VAL
        }));
    }
}

#[cfg(test)]
mod bench {
    use extra::test::BenchHarness;
    use rand::*;
    use sys::size_of;

    #[bench]
    fn rand_xorshift(bh: &mut BenchHarness) {
        let mut rng = XorShiftRng::new();
        do bh.iter {
            rng.gen::<uint>();
        }
        bh.bytes = size_of::<uint>() as u64;
    }

    #[bench]
    fn rand_isaac(bh: &mut BenchHarness) {
        let mut rng = IsaacRng::new();
        do bh.iter {
            rng.gen::<uint>();
        }
        bh.bytes = size_of::<uint>() as u64;
    }

    #[bench]
    fn rand_shuffle_100(bh: &mut BenchHarness) {
        let mut rng = XorShiftRng::new();
        let x : &mut[uint] = [1,..100];
        do bh.iter {
            rng.shuffle_mut(x);
        }
    }
}
