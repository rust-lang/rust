// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test which of the builtin types are considered sendable.

fn assert_send<T:Send>() { }

// owned content are ok
fn test30() { assert_send::<Box<int>>(); }
fn test31() { assert_send::<String>(); }
fn test32() { assert_send::<Vec<int> >(); }

// but not if they own a bad thing
fn test40<'a>(_: &'a int) {
    assert_send::<Box<&'a int>>(); //~ ERROR does not fulfill the required lifetime
}

fn main() { }
