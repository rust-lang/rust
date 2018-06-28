// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(specialization)]

// test if the intersecion impl is complete

trait A { }
trait B { }
trait C { }

trait MyTrait { }

impl<T: A> MyTrait for T { }
impl<T: B> MyTrait for T { }
//~^ ERROR conflicting implementations of trait `MyTrait`

// This would be OK:
//impl<T: A + B> MyTrait for T { }
// But what about this:
impl<T: A + B + C> MyTrait for T { }

fn main() {}
