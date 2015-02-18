// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Checks that regions which appear in a trait object type are
// observed by the variance inference algorithm (and hence
// `TOption` is contavariant w/r/t `'a` and not bivariant).
//
// Issue #18262.

#![feature(rustc_attrs)]

use std::mem;

trait T { fn foo(&self); }

#[rustc_variance]
struct TOption<'a> { //~ ERROR regions=[[-];[];[]]
    v: Option<Box<T + 'a>>,
}

fn main() { }
