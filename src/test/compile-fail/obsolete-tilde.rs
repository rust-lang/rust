// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that ~ pointers give an obsolescence message.

fn foo(x: ~int) {} //~ ERROR obsolete syntax: `~` notation for owned pointers
fn bar(x: ~str) {} //~ ERROR obsolete syntax: `~` notation for owned pointers
fn baz(x: ~[int]) {} //~ ERROR obsolete syntax: `~[T]` is no longer a type

fn main() {
    let x = ~4i; //~ ERROR obsolete syntax: `~` notation for owned pointer allocation
    let y = ~"hello"; //~ ERROR obsolete syntax: `~` notation for owned pointer allocation
    let z = ~[1i, 2, 3]; //~ ERROR obsolete syntax: `~[T]` is no longer a type
}
