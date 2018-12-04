// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// edition:2018
// aux-build:trait-import-suggestions.rs
// compile-flags:--extern trait-import-suggestions

mod foo {
    mod foobar {
        pub(crate) trait Foobar {
            fn foobar(&self) { }
        }

        impl Foobar for u32 { }
    }

    pub(crate) trait Bar {
        fn bar(&self) { }
    }

    impl Bar for u32 { }

    fn in_foo() {
        let x: u32 = 22;
        x.foobar(); //~ ERROR no method named `foobar`
    }
}

fn main() {
    let x: u32 = 22;
    x.bar(); //~ ERROR no method named `bar`
    x.baz(); //~ ERROR no method named `baz`
    let y = u32::from_str("33"); //~ ERROR no function or associated item named `from_str`
}
