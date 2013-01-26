// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// xfail-test
enum pat { pat_ident(Option<uint>) }

fn f(pat: pat) -> bool { true }

fn num_bindings(pat: pat) -> uint {
    match pat {
      pat_ident(_) if f(pat) { 0 }
      pat_ident(None) { 1 }
      pat_ident(Some(sub)) { sub }
    }
}

fn main() {}
