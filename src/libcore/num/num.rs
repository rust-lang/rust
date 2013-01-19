// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! An interface for numeric types

pub trait Num {
    // FIXME: Trait composition. (#2616)
    pure fn add(&self, other: &Self) -> Self;
    pure fn sub(&self, other: &Self) -> Self;
    pure fn mul(&self, other: &Self) -> Self;
    pure fn div(&self, other: &Self) -> Self;
    pure fn modulo(&self, other: &Self) -> Self;
    pure fn neg(&self) -> Self;

    pure fn to_int(&self) -> int;
    static pure fn from_int(n: int) -> Self;
}

pub trait IntConvertible {
    pure fn to_int(&self) -> int;
    static pure fn from_int(n: int) -> Self;
}

pub trait Zero {
    static pure fn zero() -> Self;
}

pub trait One {
    static pure fn one() -> Self;
}
