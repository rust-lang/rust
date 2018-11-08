// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// compile-pass
#![allow(warnings)]

mod foo {
    pub use foo::bar::S;
    mod bar {
        #[derive(Default)]
        pub struct S {
            pub(in foo) x: i32,
        }
        impl S {
            pub(in foo) fn f(&self) -> i32 { 0 }
        }

        pub struct S2 {
            pub(crate) x: bool,
        }
        impl S2 {
            pub(crate) fn f(&self) -> bool { false }
        }

        impl ::std::ops::Deref for S {
            type Target = S2;
            fn deref(&self) -> &S2 { unimplemented!() }
        }
    }
}


fn main() {
    let s = foo::S::default();
    let _: bool = s.x;
    let _: bool = s.f();
}
