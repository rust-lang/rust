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
    pub use self::bar::S;
    mod bar {
        pub struct S;
        pub use baz;
    }

    trait T {
        type Assoc;
    }
    impl T for () {
        type Assoc = S;
    }
}

impl foo::S {
    fn f() {}
}

pub mod baz {
    fn f() {}

    fn g() {
        ::foo::bar::baz::f(); //~ERROR module `bar` is private
        ::foo::bar::S::f(); //~ERROR module `bar` is private
        <() as ::foo::T>::Assoc::f(); //~ERROR trait `T` is private
    }
}

fn main() {}
