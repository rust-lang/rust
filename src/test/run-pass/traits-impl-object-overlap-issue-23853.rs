// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that we are able to compile the case where both a blanket impl
// and the object type itself supply the required trait obligation.
// In this case, the blanket impl for `Foo` applies to any type,
// including `Bar`, but the object type `Bar` also implicitly supplies
// this context.

trait Foo { fn dummy(&self) { } }

trait Bar: Foo { }

impl<T:?Sized> Foo for T { }

fn want_foo<B:?Sized+Foo>() { }

fn main() {
    want_foo::<Bar>();
}
