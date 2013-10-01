// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! The implementations of `Rand` for the built-in types.

use char;
use int;
use option::{Option, Some, None};
use rand::{Rand,Rng};
use u32;
use uint;

impl Rand for int {
    #[inline]
    fn rand<R: Rng>(rng: &mut R) -> int {
        if int::bits == 32 {
            rng.gen::<i32>() as int
        } else {
            rng.gen::<i64>() as int
        }
    }
}

impl Rand for i8 {
    #[inline]
    fn rand<R: Rng>(rng: &mut R) -> i8 {
        rng.next_u32() as i8
    }
}

impl Rand for i16 {
    #[inline]
    fn rand<R: Rng>(rng: &mut R) -> i16 {
        rng.next_u32() as i16
    }
}

impl Rand for i32 {
    #[inline]
    fn rand<R: Rng>(rng: &mut R) -> i32 {
        rng.next_u32() as i32
    }
}

impl Rand for i64 {
    #[inline]
    fn rand<R: Rng>(rng: &mut R) -> i64 {
        rng.next_u64() as i64
    }
}

impl Rand for uint {
    #[inline]
    fn rand<R: Rng>(rng: &mut R) -> uint {
        if uint::bits == 32 {
            rng.gen::<u32>() as uint
        } else {
            rng.gen::<u64>() as uint
        }
    }
}

impl Rand for u8 {
    #[inline]
    fn rand<R: Rng>(rng: &mut R) -> u8 {
        rng.next_u32() as u8
    }
}

impl Rand for u16 {
    #[inline]
    fn rand<R: Rng>(rng: &mut R) -> u16 {
        rng.next_u32() as u16
    }
}

impl Rand for u32 {
    #[inline]
    fn rand<R: Rng>(rng: &mut R) -> u32 {
        rng.next_u32()
    }
}

impl Rand for u64 {
    #[inline]
    fn rand<R: Rng>(rng: &mut R) -> u64 {
        rng.next_u64()
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
        let u1 = rng.next_u32() as f64;
        let u2 = rng.next_u32() as f64;
        let u3 = rng.next_u32() as f64;

        ((u1 / SCALE + u2) / SCALE + u3) / SCALE
    }
}


impl Rand for char {
    #[inline]
    fn rand<R: Rng>(rng: &mut R) -> char {
        // a char is 21 bits
        static CHAR_MASK: u32 = 0x001f_ffff;
        loop {
            // Rejection sampling. About 0.2% of numbers with at most
            // 21-bits are invalid codepoints (surrogates), so this
            // will succeed first go almost every time.
            match char::from_u32(rng.next_u32() & CHAR_MASK) {
                Some(c) => return c,
                None => {}
            }
        }
    }
}

impl Rand for bool {
    #[inline]
    fn rand<R: Rng>(rng: &mut R) -> bool {
        rng.gen::<u8>() & 1 == 1
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
