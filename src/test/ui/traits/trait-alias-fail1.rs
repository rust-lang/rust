// Copyright 2017-2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// gate-test-trait_alias

trait CloneDefault<T> = Default where T: Clone;
trait BoundedAlias<T: Clone = ()> = Default;

trait A<T: Send> {}
trait B<T> = A<T>; // FIXME: parameter T should need a bound here, or semantics should be changed

impl CloneDefault for () {}

fn main() {}
