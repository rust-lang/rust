// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Implementation of the `{:?}` format qualifier
//!
//! This module contains the `Poly` trait which is used to implement the `{:?}`
//! format expression in formatting macros. This trait is defined for all types
//! automatically, so it is likely not necessary to use this module manually

use std::fmt;

use repr;

/// Format trait for the `?` character
pub trait Poly {
    /// Formats the value using the given formatter.
    #[experimental]
    fn fmt(&self, &mut fmt::Formatter) -> fmt::Result;
}

#[doc(hidden)]
pub fn secret_poly<T: Poly>(x: &T, fmt: &mut fmt::Formatter) -> fmt::Result {
    // FIXME #11938 - UFCS would make us able call the this method
    //                directly Poly::fmt(x, fmt).
    x.fmt(fmt)
}

impl<T> Poly for T {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match (f.width, f.precision) {
            (None, None) => {
                match repr::write_repr(f, self) {
                    Ok(()) => Ok(()),
                    Err(..) => Err(fmt::WriteError),
                }
            }

            // If we have a specified width for formatting, then we have to make
            // this allocation of a new string
            _ => {
                let s = repr::repr_to_string(self);
                f.pad(s.as_slice())
            }
        }
    }
}
