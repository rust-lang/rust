// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(dead_code)]

// This test contains examples originally in `type-alias-bounds.rs`,
// that now produce errors instead of being silently accepted.

// Bounds are not checked either, i.e. the definition is not necessarily well-formed
struct Sendable<T: Send>(T);
type MySendable<T> = Sendable<T>;
//~^ ERROR `T` cannot be sent between threads safely

trait Bound { type Assoc; }
type T4<U> = <U as Bound>::Assoc;
//~^ ERROR the trait bound `U: Bound` is not satisfied

fn main() {}
