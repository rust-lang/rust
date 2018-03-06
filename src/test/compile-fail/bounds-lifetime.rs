// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

type A = for<'b, 'a: 'b> fn(); //~ ERROR lifetime bounds cannot be used in this context
type B = for<'b, 'a: 'b,> fn(); //~ ERROR lifetime bounds cannot be used in this context
type C = for<'b, 'a: 'b +> fn(); //~ ERROR lifetime bounds cannot be used in this context
type D = for<'a, T> fn(); //~ ERROR only lifetime parameters can be used in this context
type E = for<T> Fn(); //~ ERROR only lifetime parameters can be used in this context

fn main() {}
