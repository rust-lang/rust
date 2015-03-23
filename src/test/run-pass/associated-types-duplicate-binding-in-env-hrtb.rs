// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Check that we do not report ambiguities when equivalent predicates
// (modulo bound lifetime names) appears in the environment
// twice. Issue #21965.

// pretty-expanded FIXME #23616

fn foo<T>(t: T) -> i32
    where T : for<'a> Fn(&'a u8) -> i32,
          T : for<'b> Fn(&'b u8) -> i32,
{
    t(&3)
}

fn main() {
}
