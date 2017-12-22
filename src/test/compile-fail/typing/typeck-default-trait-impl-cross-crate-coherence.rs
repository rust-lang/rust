// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:tdticc_coherence_lib.rs

// Test that we do not consider associated types to be sendable without
// some applicable trait bound (and we don't ICE).

#![feature(optin_builtin_traits)]

extern crate tdticc_coherence_lib as lib;

use lib::DefaultedTrait;

struct A;
impl DefaultedTrait for (A,) { } //~ ERROR E0117

struct B;
impl !DefaultedTrait for (B,) { } //~ ERROR E0117

struct C;
struct D<T>(T);
impl DefaultedTrait for Box<C> { } //~ ERROR E0321
impl DefaultedTrait for lib::Something<C> { } //~ ERROR E0117
impl DefaultedTrait for D<C> { } // OK

fn main() { }
