// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// xfail-fast compile-flags doesn't work with fast-check
// compile-flags: --cfg foo --cfg bar(baz) --cfg qux="foo"

fn main() {
    // check
    if ! cfg!(foo) { fail2!() }
    if   cfg!(not(foo)) { fail2!() }

    if ! cfg!(bar(baz)) { fail2!() }
    if   cfg!(not(bar(baz))) { fail2!() }

    if ! cfg!(qux="foo") { fail2!() }
    if   cfg!(not(qux="foo")) { fail2!() }

    if ! cfg!(foo, bar(baz), qux="foo") { fail2!() }
    if   cfg!(not(foo, bar(baz), qux="foo")) { fail2!() }

    if cfg!(not_a_cfg) { fail2!() }
    if cfg!(not_a_cfg, foo, bar(baz), qux="foo") { fail2!() }

    if ! cfg!(not(not_a_cfg)) { fail2!() }
    if ! cfg!(not(not_a_cfg), foo, bar(baz), qux="foo") { fail2!() }

    if cfg!(trailing_comma, ) { fail2!() }
}
