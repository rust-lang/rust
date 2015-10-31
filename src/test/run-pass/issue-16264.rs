// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use outer::Foo;

mod outer {
    pub use self::inner::Foo;

    mod inner {
        pub trait Foo {
            fn bar(&self) {}
        }
        impl Foo for i32 {}
    }
}

fn main() {
    let x: i32 = 0;
    x.bar();
}
