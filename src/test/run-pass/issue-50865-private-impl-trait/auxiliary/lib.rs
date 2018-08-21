// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![crate_type = "lib"]

pub fn bar<P>( // Error won't happen if "bar" is not generic
    _baz: P,
) {
    hide_foo()();
}

fn hide_foo() -> impl Fn() { // Error won't happen if "iterate" hasn't impl Trait or has generics
    foo
}

fn foo() { // Error won't happen if "foo" isn't used in "iterate" or has generics
}
