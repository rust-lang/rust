// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

mod inner {
    pub trait Trait {
        fn f(&self) { f(); }
        fn f_ufcs(&self) { f_ufcs(); }
    }

    impl Trait for isize {}

    fn f() {}
    fn f_ufcs() {}
}

pub fn foo<T: inner::Trait>(t: T) {
    t.f();
}
pub fn foo_ufcs<T: inner::Trait>(t: T) {
    T::f_ufcs(&t);
}
