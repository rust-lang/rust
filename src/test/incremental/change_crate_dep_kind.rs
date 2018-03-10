// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that we detect changes to the `dep_kind` query. If the change is not
// detected then -Zincremental-verify-ich will trigger an assertion.

// revisions:cfail1 cfail2
// compile-flags: -Z query-dep-graph -Cpanic=unwind
// must-compile-successfully

#![feature(panic_unwind)]

// Turn the panic_unwind crate from an explicit into an implicit query:
#[cfg(cfail1)]
extern crate panic_unwind;

fn main() {}
