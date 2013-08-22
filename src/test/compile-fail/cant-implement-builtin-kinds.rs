// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// See issue #8517 for why this should be illegal.

struct X<T>(T);

impl <T> Send for X<T> { } //~ ERROR cannot provide an explicit implementation for a builtin kind
impl <T> Sized for X<T> { } //~ ERROR cannot provide an explicit implementation for a builtin kind
impl <T> Freeze for X<T> { } //~ ERROR cannot provide an explicit implementation for a builtin kind

fn main() { }
