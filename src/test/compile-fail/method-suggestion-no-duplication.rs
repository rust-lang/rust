// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// issue #21405

fn foo<F>(f: F) where F: FnMut(usize) {}

fn main() {
    foo(|s| s.is_empty());
    //~^ ERROR does not implement any method
    //~^^ HELP #1: `core::slice::SliceExt`
    //~^^^ HELP #2: `core::str::StrExt`
    //~^^^^ HELP #3: `collections::slice::SliceExt`
    //~^^^^^ HELP #4: `collections::str::StrExt`
}
