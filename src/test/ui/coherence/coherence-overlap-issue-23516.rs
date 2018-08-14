// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Tests that we consider `Box<U>: !Sugar` to be ambiguous, even
// though we see no impl of `Sugar` for `Box`. Therefore, an overlap
// error is reported for the following pair of impls (#23516).

pub trait Sugar { fn dummy(&self) { } }
pub trait Sweet { fn dummy(&self) { } }
impl<T:Sugar> Sweet for T { }
impl<U:Sugar> Sweet for Box<U> { }
//~^ ERROR E0119

fn main() { }
