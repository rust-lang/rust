// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

mod foo {
    pub struct Foo;
    impl Foo {
        fn bar(&self) {}
    }

    pub trait Baz {
        fn bar(&self) -> bool { true }
    }
    impl Baz for Foo {}
}

fn main() {
    use foo::Baz;

    // Check that `bar` resolves to the trait method, not the inherent impl method.
    let _: () = foo::Foo.bar(); //~ ERROR mismatched types
}
