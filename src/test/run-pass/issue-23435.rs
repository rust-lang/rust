// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that we do not ICE when a default method implementation has
// requirements (in this case, `Self : Baz`) that do not hold for some
// specific impl (in this case, `Foo : Bar`). This causes problems
// only when building a vtable, because that goes along and
// instantiates all the methods, even those that could not otherwise
// be called.

// pretty-expanded FIXME #23616

struct Foo {
    x: i32
}

trait Bar {
    fn bar(&self) where Self : Baz { self.baz(); }
}

trait Baz {
    fn baz(&self);
}

impl Bar for Foo {
}

fn main() {
    let x: &Bar = &Foo { x: 22 };
}
