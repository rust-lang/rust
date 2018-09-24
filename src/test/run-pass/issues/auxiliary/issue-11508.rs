// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

pub struct Closed01<F>(pub F);

pub trait Bar { fn new() -> Self; }

impl<T: Bar> Bar for Closed01<T> {
    fn new() -> Closed01<T> { Closed01(Bar::new()) }
}
impl Bar for f32 { fn new() -> f32 { 1.0 } }

pub fn random<T: Bar>() -> T { Bar::new() }
