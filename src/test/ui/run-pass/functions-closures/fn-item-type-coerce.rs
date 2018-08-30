// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test implicit coercions from a fn item type to a fn pointer type.

// pretty-expanded FIXME #23616

fn foo(x: isize) -> isize { x * 2 }
fn bar(x: isize) -> isize { x * 4 }
type IntMap = fn(isize) -> isize;

fn eq<T>(x: T, y: T) { }

fn main() {
    let f: IntMap = foo;

    eq::<IntMap>(foo, bar);
}
