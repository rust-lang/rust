// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Issue #3902. We are (at least currently) unable to infer `Self`
// based on `T`, even though there is only a single impl, because of
// the possibility of associated types and other things (basically: no
// constraints on `Self` here at all).

mod base {
    pub trait HasNew<T> {
        fn new() -> T;
        fn dummy(&self) { }
    }

    pub struct Foo {
        dummy: (),
    }

    impl HasNew<Foo> for Foo {
        fn new() -> Foo {
            Foo { dummy: () }
        }
    }
}

pub fn foo() {
    let _f: base::Foo = base::HasNew::new();
    //~^ ERROR type annotations required
}

fn main() { }
