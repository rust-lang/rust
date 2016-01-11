// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test of allowing two sequences repetitions in a row,
// functionality added as byproduct of RFC amendment #1384
//   https://github.com/rust-lang/rfcs/pull/1384

// Old version of Rust would reject this macro definition, even though
// there are no local ambiguities (the initial `banana` and `orange`
// tokens are enough for the expander to distinguish which case is
// intended).
macro_rules! foo {
    ( $(banana $a:ident)* $(orange $b:tt)* ) => { };
}

fn main() {
    foo!( banana id1 banana id2
          orange hi  orange (hello world) );
}
