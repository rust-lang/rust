// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Coherence error results because we do not know whether `T: Foo<P>` or not
// for the second impl.

use std::marker::PhantomData;

pub trait Foo<P> {}

impl <P, T: Foo<P>> Foo<P> for Option<T> {} //~ ERROR E0119

impl<T, U> Foo<T> for Option<U> { }

fn main() {}
