// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// pretty-expanded FIXME #23616

struct Foo;

trait Trait {
    fn bar(&self);
}

// Inherent impls should be preferred over trait ones.
impl Foo {
    fn bar(&self) {}
}

impl Trait {
    fn baz(_: &Foo) {}
}

impl Trait for Foo {
    fn bar(&self) { panic!("wrong method called!") }
}

fn main() {
    Foo.bar();
    Foo::bar(&Foo);
    <Foo>::bar(&Foo);

    // Should work even if Trait::baz doesn't exist.
    // N.B: `<Trait>::bar` would be ambiguous.
    <Trait>::baz(&Foo);
}
