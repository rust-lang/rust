// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// compile-flags: --cfg fooA --cfg fooB

#[cfg(fooA, not(bar))] // fooA AND !bar
fn foo1() -> int { 1 }

#[cfg(not(fooA, bar))] // !fooA AND !bar
fn foo2() -> int { 2 }

#[cfg(fooC)]
#[cfg(fooB, not(bar))] // fooB AND !bar
fn foo2() -> int { 3 } // fooC OR (fooB AND !bar)


fn main() {
    fail_unless!(1 == foo1());
    fail_unless!(3 == foo2());
}
