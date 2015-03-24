// Copyright 2013-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// compile-flags: --cfg foo --cfg qux="foo"

// pretty-expanded FIXME #23616

pub fn main() {
    // check
    if ! cfg!(foo) { panic!() }
    if   cfg!(not(foo)) { panic!() }

    if ! cfg!(qux="foo") { panic!() }
    if   cfg!(not(qux="foo")) { panic!() }

    if ! cfg!(all(foo, qux="foo")) { panic!() }
    if   cfg!(not(all(foo, qux="foo"))) { panic!() }
    if   cfg!(all(not(all(foo, qux="foo")))) { panic!() }

    if cfg!(not_a_cfg) { panic!() }
    if cfg!(all(not_a_cfg, foo, qux="foo")) { panic!() }
    if cfg!(all(not_a_cfg, foo, qux="foo")) { panic!() }
    if ! cfg!(any(not_a_cfg, foo)) { panic!() }

    if ! cfg!(not(not_a_cfg)) { panic!() }
    if ! cfg!(all(not(not_a_cfg), foo, qux="foo")) { panic!() }
}
