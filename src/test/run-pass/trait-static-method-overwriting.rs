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
        static pure fn new() -> T;
    }

    pub struct Foo {
        dummy: (),
    }

    pub impl Foo : base::HasNew<Foo> {
        static pure fn new() -> Foo {
			unsafe { io::println("Foo"); }
            Foo { dummy: () }
        }
    }

    pub struct Bar {
        dummy: (),
    }

    pub impl Bar : base::HasNew<Bar> {
        static pure fn new() -> Bar {
			unsafe { io::println("Bar"); }
            Bar { dummy: () }
        }
    }
}

fn main() {
    let f: base::Foo = base::new::<base::Foo, base::Foo>();
	let b: base::Bar = base::new::<base::Bar, base::Bar>();
}
