// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// check that we don't accidentally capture upvars just because their name
// occurs in a path

fn assert_static<T: 'static>(_t: T) {}

mod foo {
    pub fn scope() {}
}

fn main() {
    let scope = &mut 0;
    assert_static(|| {
       foo::scope();
    });
}
