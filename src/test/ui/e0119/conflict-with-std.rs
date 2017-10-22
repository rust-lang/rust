// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(try_from)]

use std::marker::PhantomData;
use std::convert::{TryFrom, AsRef};

struct Q;
impl AsRef<Q> for Box<Q> {
    fn as_ref(&self) -> &Q {
        &**self
    }
}

struct S;
impl From<S> for S {
    fn from(s: S) -> S {
        s
    }
}

struct X;
impl TryFrom<X> for X {
    type Error = ();
    fn try_from(u: X) -> Result<X, ()> {
        Ok(u)
    }
}

fn main() {}