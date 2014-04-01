// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Ensure that moves out of statics don't null the static

use std::kinds::marker;

struct Foo {
    foo: int,
    nocopy: marker::NoCopy
}

impl Eq for Foo {
    fn eq(&self, other: &Foo) -> bool {
        self.foo == other.foo
    }
}

impl std::fmt::Show for Foo {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f.buf, "Foo({})", self.foo)
    }
}

static BAR: Foo = Foo{foo: 5, nocopy: marker::NoCopy};

fn main() {
    let x = BAR;
    assert_eq!(x, BAR);
}
