// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Make sure that the mono-item collector does not crash when trying to
// instantiate a default impl of a method with lifetime parameters.
// See https://github.com/rust-lang/rust/issues/47309

// compile-flags:-Clink-dead-code
// compile-pass

#![crate_type="rlib"]

pub trait EnvFuture {
    type Item;

    fn boxed_result<'a>(self) where Self: Sized, Self::Item: 'a, {
    }
}

struct Foo;

impl<'a> EnvFuture for &'a Foo {
    type Item = ();
}
