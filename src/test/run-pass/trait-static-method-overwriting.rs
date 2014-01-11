// xfail-fast

// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

mod base {
    pub trait HasNew<T> {
        fn new() -> Self;
    }

    pub struct Foo {
        dummy: (),
    }

    impl ::base::HasNew<Foo> for Foo {
        fn new() -> Foo {
            println!("Foo");
            Foo { dummy: () }
        }
    }

    pub struct Bar {
        dummy: (),
    }

    impl ::base::HasNew<Bar> for Bar {
        fn new() -> Bar {
            println!("Bar");
            Bar { dummy: () }
        }
    }
}

pub fn main() {
    let _f: base::Foo = base::HasNew::<base::Foo>::new();
    let _b: base::Bar = base::HasNew::<base::Bar>::new();
}
