// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Basic test for traits inheriting from the builtin kinds, checking
// the type contents of the implementing type (that's not a typaram).

trait Foo : Send { }

impl <'self> Foo for &'self mut () { } //~ ERROR cannot implement this trait

trait Bar : Freeze { }

impl <'self> Bar for &'self mut () { } //~ ERROR cannot implement this trait

fn main() { }
