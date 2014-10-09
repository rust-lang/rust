// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Simple numerics.
//!
//! This crate contains arbitrary-sized integer, rational, and complex types.
//!
//! ## Example
//!
//! This example uses the BigRational type and [Newton's method][newt] to
//! approximate a square root to arbitrary precision:
//!
//! ```
//! # #![allow(deprecated)]
//! extern crate num;
//!
//! use num::bigint::BigInt;
//! use num::rational::{Ratio, BigRational};
//!
//! fn approx_sqrt(number: u64, iterations: uint) -> BigRational {
//!     let start: Ratio<BigInt> = Ratio::from_integer(FromPrimitive::from_u64(number).unwrap());
//!     let mut approx = start.clone();
//!
//!     for _ in range(0, iterations) {
//!         approx = (approx + (start / approx)) /
//!             Ratio::from_integer(FromPrimitive::from_u64(2).unwrap());
//!     }
//!
//!     approx
//! }
//!
//! fn main() {
//!     println!("{}", approx_sqrt(10, 4)); // prints 4057691201/1283082416
//! }
//! ```
//!
//! [newt]: https://en.wikipedia.org/wiki/Methods_of_computing_square_roots#Babylonian_method

#![allow(unknown_features)]
#![feature(macro_rules, slicing_syntax)]
#![feature(default_type_params)]

#![crate_name = "num"]
#![deprecated = "This is now a cargo package located at: \
                 https://github.com/rust-lang/num"]
#![allow(deprecated)]
#![crate_type = "rlib"]
#![crate_type = "dylib"]
#![license = "MIT/ASL2"]
#![doc(html_logo_url = "http://www.rust-lang.org/logos/rust-logo-128x128-blk-v2.png",
       html_favicon_url = "http://www.rust-lang.org/favicon.ico",
       html_root_url = "http://doc.rust-lang.org/nightly/",
       html_playground_url = "http://play.rust-lang.org/")]
#![allow(deprecated)] // from_str_radix

extern crate rand;

pub use bigint::{BigInt, BigUint};
pub use rational::{Rational, BigRational};
pub use complex::Complex;
pub use integer::Integer;

pub mod bigint;
pub mod complex;
pub mod integer;
pub mod rational;
