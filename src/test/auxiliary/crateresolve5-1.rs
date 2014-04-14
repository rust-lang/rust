// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![crate_id="crateresolve5#0.1"]

#![crate_type = "lib"]

pub struct NameVal { pub name: ~str, pub val: int }

pub fn struct_nameval() -> NameVal {
    NameVal { name: ~"crateresolve5", val: 10 }
}

pub enum e {
    e_val
}

pub fn nominal() -> e { e_val }

pub fn nominal_eq(_e1: e, _e2: e) -> bool { true }

impl Eq for e {
    fn eq(&self, other: &e) -> bool { nominal_eq(*self, *other) }
    fn ne(&self, other: &e) -> bool { !nominal_eq(*self, *other) }
}

pub fn f() -> int { 10 }
