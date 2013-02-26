// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/*!
 * Try to double-check that const fns have the right size (with or
 * without dummy env ptr, as appropriate) by iterating a size-2 array.
 * If the const size differs from the runtime size, the second element
 * should be read as a null or otherwise wrong pointer and crash.
 */

fn f() { }
const bare_fns: &'static [extern fn()] = &[f, f];
struct S<'self>(&'self fn());
const closures: &'static [S<'static>] = &[S(f), S(f)];

pub fn main() {
    for bare_fns.each |&bare_fn| { bare_fn() }
    for closures.each |&closure| { (*closure)() }
}
