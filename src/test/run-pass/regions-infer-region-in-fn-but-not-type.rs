// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


// check that the &int here does not cause us to think that `foo`
// contains region pointers
struct foo(Box<FnMut(&int)+'static>);

fn take_foo<T:'static>(x: T) {}

fn have_foo(f: foo) {
    take_foo(f);
}

fn main() {}
