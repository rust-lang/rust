// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

struct Foo(bool);
const NEW_FALSE: bool = false;
const STATIC_FOO: Foo = Foo(NEW_FALSE);

pub fn main() {
    match (Foo(false)) {
        STATIC_FOO => 3,
        _ => 11
    };
}
