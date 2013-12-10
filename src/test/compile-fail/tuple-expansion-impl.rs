// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// error-pattern: cannot expand non-tuple type `..T`

trait Tuple {}

// FIXME(eddyb) #10769 these generic substitution errors don't have spans.
impl<T> Tuple for (..T) {}
//^ ERROR cannot determine a type for this bounded type parameter: unconstrained type

fn main() {}
