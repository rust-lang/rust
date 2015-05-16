// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Regression test for issue #25436: check that things which can be
// followed by any token also permit X* to come afterwards.

macro_rules! foo {
  ( $a:tt $($b:tt)* ) => { };
  ( $a:ident $($b:tt)* ) => { };
  ( $a:item $($b:tt)* ) => { };
  ( $a:block $($b:tt)* ) => { };
  ( $a:meta $($b:tt)* ) => { }
}

fn main() { }
