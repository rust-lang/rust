// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Checks for private types in public interfaces

mod y {
    pub struct Foo { x: u32 }

    struct Bar { x: u32 }

    impl Foo {
        pub fn foo(&self, x: Self, y: Bar) { } //~ ERROR private type in public interface
    }
}

mod x {
    pub struct Foo { pub x: u32 }

    struct Bar { _x: u32 }

    impl Foo {
        pub fn foo(&self, _x: Self, _y: Bar) { } //~ ERROR private type in public interface
        pub fn bar(&self) -> Bar { Bar { _x: self.x } }
        //~^ ERROR private type in public interface
    }
}

pub fn main() {
    let f = x::Foo { x: 4 };
    let b = f.bar();
    f.foo(x::Foo { x: 5 }, b);
}
